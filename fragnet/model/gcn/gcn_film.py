from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_softmax
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add


# ─────────────────────────── FiLM 调制模块 ───────────────────────────

class FiLMGenerator(nn.Module):
    """
    给定条件特征 c（形状 [N, cond_dim]），生成 gamma 和 beta（形状 [N, feat_dim]）
    用于对节点/片段表示做 FiLM 调制：gamma * h + beta
    """
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feat_dim)
        self.beta_net  = nn.Linear(cond_dim, feat_dim)
        # gamma 初始化为 1（不改变尺度），beta 初始化为 0（不偏移）
        nn.init.ones_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        c: [N, cond_dim]  条件特征（每个节点/片段对应一行）
        h: [N, feat_dim]  待调制的表示
        return: [N, feat_dim]
        """
        gamma = self.gamma_net(c)   # [N, feat_dim]
        beta  = self.beta_net(c)    # [N, feat_dim]
        return gamma * h + beta


# ─────────────────────────── FragNetLayer（带 FiLM）───────────────────────────

class FragNetLayer(nn.Module):
    def __init__(self, atom_in=128, atom_out=128, frag_in=128, frag_out=128,
                 edge_in=128, edge_out=128,
                 cond_dim: int = 0):          # ← 新增：条件特征维度，0 表示不用 FiLM
        super(FragNetLayer, self).__init__()

        self.atom_embed = nn.Linear(atom_in, atom_out, bias=True)
        self.frag_embed = nn.Linear(frag_in, frag_out)
        self.edge_embed = nn.Linear(edge_in, edge_out)

        self.frag_message_mlp = nn.Linear(atom_out*2, atom_out)
        self.atom_mlp = torch.nn.Sequential(torch.nn.Linear(atom_out, 2*atom_out),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(2*atom_out, atom_out))

        self.frag_mlp = torch.nn.Sequential(torch.nn.Linear(atom_out, 2*atom_out),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(2*atom_out, atom_out))

        # FiLM 调制器（可选）
        self.use_film = cond_dim > 0
        if self.use_film:
            self.film_atom = FiLMGenerator(cond_dim, atom_out)
            self.film_frag = FiLMGenerator(cond_dim, atom_out)

    def forward(self, x_atoms, edge_index, edge_attr,
                frag_index, x_frags, atom_to_frag_ids,
                cond_atom=None, cond_frag=None):
        """
        新增参数：
          cond_atom: [num_atoms, cond_dim]  每个原子对应的条件特征
          cond_frag: [num_frags, cond_dim]  每个片段对应的条件特征
        """
        edge_index, _ = add_self_loops(edge_index=edge_index)

        self_loop_attr = torch.zeros(x_atoms.size(0), edge_attr.size(1), dtype=torch.long)
        self_loop_attr[:, 0] = 0
        edge_attr = torch.cat((edge_attr, self_loop_attr.to(edge_attr)), dim=0)

        x_atoms = self.atom_embed(x_atoms)
        edge_attr = self.edge_embed(edge_attr)

        source, target = edge_index
        source_features = torch.index_select(input=x_atoms, index=source, dim=0)

        deg = degree(source, x_atoms.size(0), dtype=x_atoms.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]

        message = source_features * norm.view(-1, 1)
        x_atoms_new = scatter_add(src=message, index=target, dim=0)

        # ── FiLM 调制原子表示 ──
        if self.use_film and cond_atom is not None:
            x_atoms_new = self.film_atom(cond_atom, x_atoms_new)

        x_frags = scatter_add(src=x_atoms_new, index=atom_to_frag_ids, dim=0)

        source, target = frag_index
        source_features = torch.index_select(input=x_frags, index=source, dim=0)
        frag_message = source_features
        frag_feats_sum = scatter_add(src=frag_message, index=target, dim=0)
        frag_feats_sum = self.frag_mlp(frag_feats_sum)

        # ── FiLM 调制片段表示 ──
        if self.use_film and cond_frag is not None:
            frag_feats_sum = self.film_frag(cond_frag, frag_feats_sum)

        x_frags_new = frag_feats_sum
        return x_atoms_new, x_frags_new


# ─────────────────────────── FragNet（带 FiLM）───────────────────────────

class FragNet(nn.Module):

    def __init__(self, num_layer, drop_ratio=0, emb_dim=128,
                 atom_features=167, frag_features=167, edge_features=17,
                 cond_dim: int = 0):          # ← 新增
        super(FragNet, self).__init__()
        self.num_layer = num_layer
        self.dropout = nn.Dropout(p=drop_ratio)
        self.act = nn.ReLU()

        layer_kwargs = dict(
            atom_out=emb_dim, frag_out=emb_dim,
            edge_in=edge_features, edge_out=emb_dim,
            cond_dim=cond_dim,
        )

        self.layer1 = FragNetLayer(atom_in=atom_features, frag_in=frag_features, **layer_kwargs)
        self.layer2 = FragNetLayer(atom_in=emb_dim,       frag_in=emb_dim,       **layer_kwargs)
        self.layer3 = FragNetLayer(atom_in=emb_dim,       frag_in=emb_dim,       **layer_kwargs)
        self.layer4 = FragNetLayer(atom_in=emb_dim,       frag_in=emb_dim,       **layer_kwargs)

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        # 条件特征投影：将分子级 cond 广播到原子/片段级
        self.use_film = cond_dim > 0
        if self.use_film:
            # 如果输入的 cond 是分子级（1行/分子），需要先投影
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.ReLU(),
            )

    def forward(self, batch):
        x_atoms          = batch['x_atoms']
        edge_index        = batch['edge_index']
        frag_index        = batch['frag_index']
        x_frags           = batch['x_frags']
        edge_attr         = batch['edge_attr']
        atom_to_frag_ids  = batch['atom_to_frag_ids']

        # ── 准备条件特征 ──
        # cond_mol: [num_molecules, cond_dim]，通过 batch 索引广播到原子/片段
        cond_atom = cond_frag = None
        if self.use_film and 'cond' in batch:
            cond_mol  = self.cond_proj(batch['cond'].float())   # [num_mol, cond_dim]
            # batch['batch']:      [num_atoms]，值为该原子所属分子的索引
            # batch['frag_batch']: [num_frags]，值为该片段所属分子的索引
            cond_atom = cond_mol[batch['batch']]                # [num_atoms, cond_dim]
            cond_frag = cond_mol[batch['frag_batch']]           # [num_frags, cond_dim]

        x_atoms = self.dropout(x_atoms)
        x_frags = self.dropout(x_frags)

        x_atoms, x_frags = self.layer1(x_atoms, edge_index, edge_attr,
                                       frag_index, x_frags, atom_to_frag_ids,
                                       cond_atom, cond_frag)
        x_atoms, x_frags = self.act(x_atoms), self.act(x_frags)

        x_atoms, x_frags = self.layer2(x_atoms, edge_index, edge_attr,
                                       frag_index, x_frags, atom_to_frag_ids,
                                       cond_atom, cond_frag)
        x_atoms, x_frags = self.act(x_atoms), self.act(x_frags)

        x_atoms, x_frags = self.layer3(x_atoms, edge_index, edge_attr,
                                       frag_index, x_frags, atom_to_frag_ids,
                                       cond_atom, cond_frag)
        x_atoms, x_frags = self.act(x_atoms), self.act(x_frags)

        x_atoms, x_frags = self.layer4(x_atoms, edge_index, edge_attr,
                                       frag_index, x_frags, atom_to_frag_ids,
                                       cond_atom, cond_frag)
        x_atoms, x_frags = self.act(x_atoms), self.act(x_frags)

        return x_atoms, x_frags


class FragNetPreTrain(nn.Module):

    def __init__(self, atom_features=167, frag_features=167, edge_features=17,
                emb_dim=128, num_layer=6, drop_ratio=0.15, cond_dim=0):
        super(FragNetPreTrain, self).__init__()

        self.pretrain = FragNet(
            num_layer=num_layer,
            drop_ratio=drop_ratio,
            emb_dim=emb_dim,
            atom_features=atom_features,
            frag_features=frag_features,
            edge_features=edge_features,
            cond_dim=cond_dim,             # ← 透传
        )
        self.lin1 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 118)
        self.dropout = nn.Dropout(p=0.15)
        self.activation = nn.ReLU()

    def forward(self, batch):
        x_atoms, x_frags = self.pretrain(batch)

        x = self.dropout(x_atoms)
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


class FragNetFineTune(nn.Module):

    def __init__(self, n_classes=1,
                 atom_features=167, frag_features=167, edge_features=17,
                 emb_dim=128, drop_ratio=0.15, num_layer=6,
                 cond_dim=0):              # ← 新增
        super(FragNetFineTune, self).__init__()

        self.pretrain = FragNet(
            num_layer=num_layer,
            drop_ratio=drop_ratio,
            emb_dim=emb_dim,
            atom_features=atom_features,
            frag_features=frag_features,
            edge_features=edge_features,
            cond_dim=cond_dim,             # ← 透传
        )
        self.lin1 = nn.Linear(emb_dim * 2, emb_dim * 2)
        self.dropout = nn.Dropout(p=drop_ratio)
        self.activation = nn.ReLU()
        self.out = nn.Linear(emb_dim * 2, n_classes)

    def forward(self, batch):
        x_atoms, x_frags = self.pretrain(batch)

        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)

        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        x = self.dropout(cat)
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        return x