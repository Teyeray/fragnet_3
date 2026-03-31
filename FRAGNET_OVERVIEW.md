# FragNet 完整技术文档

> **FragNet: A Graph Neural Network with Four Layers of Interpretability**
> 作者: Gihan Panapitiya (PNNL / Battelle Memorial Institute)
> 框架: PyTorch + PyTorch Geometric + RDKit

---

## 目录

1. [项目概述](#1-项目概述)
2. [依赖与安装](#2-依赖与安装)
3. [数据预处理](#3-数据预处理)
   - 3.1 分子片段分解
   - 3.2 特征提取
   - 3.3 四图构建
   - 3.4 批处理 (collate_fn)
4. [网络结构](#4-网络结构)
   - 4.1 FragNetLayerA（注意力层）
   - 4.2 FragNet（骨干网络）
   - 4.3 回归/分类头
   - 4.4 FragNetFineTune（下游任务模型）
5. [预训练](#5-预训练)
   - 5.1 预训练任务与目标
   - 5.2 PretrainTask 头部
   - 5.3 训练流程
6. [微调（Fine-tuning）](#6-微调fine-tuning)
   - 6.1 配置系统
   - 6.2 训练循环
   - 6.3 超参数搜索
7. [可解释性（Interpretability）](#7-可解释性interpretability)
   - 7.1 注意力权重提取
   - 7.2 掩码归因（Masking Attribution）
   - 7.3 可视化系统
   - 7.4 交互式 Web App
8. [整体数据流](#8-整体数据流)
9. [支持的任务与数据集](#9-支持的任务与数据集)
10. [实验配置示例](#10-实验配置示例)

---

## 1. 项目概述

FragNet 是一个专为**分子性质预测**设计的图神经网络，核心创新在于**四层次可解释性**架构：

| 层次 | 实体 | 描述 |
|------|------|------|
| Level 1 | **原子 (Atoms)** | 分子中每个重原子的特征与注意力权重 |
| Level 2 | **键 (Bonds)** | 原子图中的边转化为独立节点，形成键图 |
| Level 3 | **片段 (Fragments)** | 通过 BRICS/Murcko 分解得到的子结构 |
| Level 4 | **片段间连接 (Inter-fragment bonds)** | 片段之间的共价/非共价连接 |

每一层都有独立的注意力机制，使得模型可以在**原子、键、片段、片段连接**四个粒度上提供解释。

**应用场景：**
- 分子溶解度预测 (logS)
- 脂溶性预测 (Lipophilicity)
- 分子能量预测
- 药物-靶点相互作用 (DTA)
- 癌症药物反应预测 (CDRP)

---

## 2. 依赖与安装

```bash
# 核心依赖
torch==2.4.0
torch-geometric==2.6.1
torch-scatter
rdkit==2023.9.6
networkx==2.8.8
pandas==2.2.3
numpy==1.24.1
scikit-learn==1.3.2
omegaconf==2.3.0    # YAML配置管理
optuna==3.5.0       # 超参数优化
streamlit==1.38.0   # Web可视化
lmdb==1.4.1

# 安装
pip install -e .
```

---

## 3. 数据预处理

### 3.1 分子片段分解

**文件**: `fragnet/dataset/fragments.py`

FragNet 使用 RDKit 的 **BRICS** 或 **Murcko** 方法将分子分解为片段。

#### 3D 坐标生成

```python
def get_3Dcoords(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    seed = 42
    res = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if res == 0:
        AllChem.MMFFOptimizeMolecule(mol)  # MMFF94 力场优化
    elif res == -1:
        # 增加尝试次数
        AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
        mol = AllChem.AddHs(mol, addCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)
    return mol  # 失败时退回到2D坐标
```

#### FragmentedMol 核心类

```python
class FragmentedMol:
    def __init__(self, mol, conf, frag_type="brics"):
        Chem.WedgeMolBonds(mol, conf)  # 处理立体化学
        self.mol = mol

        if frag_type == "brics":
            frag_bonds = list(BRICS.FindBRICSBonds(mol))  # 逆合成相关键
            frag_bonds = [b[0] for b in frag_bonds]
        elif frag_type == "murcko":
            frag_bonds = find_murcko_link_bond(mol)       # Murcko骨架连接键

        # 断开片段间的键
        rwmol = Chem.RWMol(mol)
        for atom_idx1, atom_idx2 in frag_bonds:
            remove_bond(rwmol, atom_idx1, atom_idx2)

        broken_mol = rwmol.GetMol()
        atomMap = Chem.GetMolFrags(broken_mol)  # 获取每个片段的原子索引

        # 构建 Fragment 对象
        fragments = [Fragment(self, atom_indices, FragIdx=i)
                     for i, atom_indices in enumerate(atomMap)]
        self.fragments = fragments

        # 构建片段间 Connection 对象
        connections = []
        for (atom_id1, atom_id2) in frag_bonds:
            bond = mol.GetBondBetweenAtoms(atom_id1, atom_id2)
            frag1 = fragment_map[atom_id1]
            frag2 = fragment_map[atom_id2]
            connection = Connection(frag1, frag2, atom_id1, atom_id2,
                                    bond.GetIdx(), bond.GetBondType(), bond)
            connections.append(connection)

        # 孤立片段（多组分分子中不共价连接的片段）特殊处理
        if len(Chem.GetMolFrags(self.mol)) > 1:
            sg_frags = self.get_atoms_in_molfrags()
            new_connections = self.add_connections_bw_molfrags(sg_frags, EmptyBond())
            connections += new_connections  # 标记为 "iso_cn3" 类型

        self.connections = tuple(connections)
```

**特殊连接类型：**
- `self_cn`：分子只有一个片段（未被分解）时的自环
- `iso_cn3`：多组分分子中不同组分片段之间的非共价连接

### 3.2 特征提取

**文件**: `fragnet/dataset/features.py`

#### 原子特征向量 (167 维)

```python
def atom_features_one_hot(self, atom):
    atom_type  = one_of_k_encoding_unk(atom.GetAtomicNum(), list(range(1,119)))  # 118维
    degree     = one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10])  # 11维
    valence    = one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6]) # 7维
    charge     = one_of_k_encoding_unk(atom.GetFormalCharge(), [-5,-4,-3,-2,-1,0,1,2,3,4,5]) # 11维
    rad_elec   = one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0,1,2,3,4])  # 5维
    hyb        = one_of_k_encoding_unk(atom.GetHybridization(),
                    [S, SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED])                   # 7维
    arom       = one_of_k_encoding(atom.GetIsAromatic(), [False, True])            # 2维
    atom_ring  = one_of_k_encoding(atom.IsInRing(), [False, True])                 # 2维
    chiral     = one_of_k_encoding_unk(atom.GetChiralTag(), [CW, CCW, UNSPECIFIED])# 3维
    numhs      = [atom.GetTotalNumHs()]                                            # 1维
    # 合计: 118+11+7+11+5+7+2+2+3+1 = 167维
    return np.array(atom_type + degree + valence + charge + rad_elec +
                    hyb + arom + atom_ring + chiral + numhs)
```

#### 键特征向量 (17 维)

```python
def bond_features_one_hot(self, bond, use_chirality=True):
    bond_type = [SINGLE, DOUBLE, TRIPLE, AROMATIC]                  # 4维 (bool)
    conj      = one_of_k_encoding(bond.GetIsConjugated(), [F, T])   # 2维
    inring    = one_of_k_encoding(bond.IsInRing(), [F, T])          # 2维
    stereo    = one_of_k_encoding_unk(str(bond.GetStereo()),
                    ["STEREOANY","STEREOZ","STEREOE","STEREONONE"])  # 4维
    bonddir   = one_of_k_encoding_unk(bond.GetBondDir(),
                    [BEGINWEDGE,BEGINDASH,ENDDOWNRIGHT,ENDUPRIGHT,NONE]) # 5维
    # 合计: 4+2+2+4+5 = 17维
```

#### 片段间连接特征 (6 维)

```python
def connection_features_one_hot(self, connection):
    bond_feats = [
        bt == SINGLE,    # 共价单键连接
        bt == DOUBLE,    # 共价双键连接
        bt == TRIPLE,    # 共价三键连接
        bt == AROMATIC,  # 芳香键连接
        bt == "self_cn", # 自环（单一片段分子）
        bt == "iso_cn3", # 非共价连接（多组分分子）
    ]  # 共 6 维
```

### 3.3 四图构建

**文件**: `fragnet/dataset/data.py`

FragNet 从每个分子构建 **4 种图**：

```
原子图 (Atom Graph)
├── 节点: 所有重原子 (167维特征)
└── 边: 分子共价键 (17维特征, 双向)

键图 (Bond Graph)
├── 节点: 原子图中的每条边变为节点 (17维特征)
└── 边: 共享同一原子的两条键之间的连接

片段图 (Fragment Graph)
├── 节点: BRICS/Murcko 分解后的片段 (167维, 片段内原子特征之和)
└── 边: 片段间的 Connection 对象 (6维特征)

片段键图 (Fragment-Bond Graph)
├── 节点: 片段图中的边（即片段间连接）变为节点
└── 边: 共享同一片段的两条片段连接之间的连接
```

**片段特征** 通过聚合片段内所有原子特征得到：
```python
x_frags[frag_id] = sum(x_atoms[atom_id] for atom_id in frag.atom_indices)
```

### 3.4 批处理 (collate_fn)

处理不同大小分子的批次：

```python
def collate_fn(data_list):
    # 拼接原子特征
    x_atoms_batch = torch.cat([i.x_atoms for i in data_list], dim=0)

    # 边索引需要偏移，以适应批次中的全局节点编号
    edge_index = torch.cat([i.edge_index for i in data_list], dim=1)
    incr_atom_nodes = get_incr_atom_nodes(data_list)  # 累积原子数偏移
    edge_index = edge_index + incr_atom_nodes

    # 同样处理片段图、键图、片段键图
    frag_index = frag_index + get_incr_frag_nodes(data_list)
    edge_index_bonds_graph = edge_index_bonds_graph + get_incr_bond_nodes(data_list)
    edge_index_fragbonds = edge_index_fragbonds + get_incr_fbond_nodes(data_list)

    # batch 向量：记录每个节点属于哪个图
    batch = torch.cat([torch.zeros(data_list[i].x_atoms.shape[0]) + i
                       for i in range(len(data_list))])
    frag_batch = torch.cat([torch.zeros(data_list[i].n_frags.item()) + i
                            for i in range(len(data_list))])

    return {
        'x_atoms': x_atoms_batch,          # (总原子数, 167)
        'edge_index': edge_index,           # (2, 总键数*2)
        'edge_attr': edge_attr,             # (总键数*2, 17)
        'x_frags': x_frags,                # (总片段数, 167)
        'frag_index': frag_index,           # (2, 总连接数)
        'edge_attr_fbonds': edge_attr_fragbonds,  # (总片段连接数, 6)
        'node_features_bonds': node_features_bonds,
        'edge_index_bonds_graph': edge_index_bonds_graph,
        'edge_attr_bonds': edge_attr_bonds,
        'node_features_fbonds': node_features_fragbonds,
        'edge_index_fbonds': edge_index_fragbonds,
        'batch': batch,         # 原子批次索引
        'frag_batch': frag_batch,  # 片段批次索引
        'atom_to_frag_ids': atom_to_frag_ids,  # 原子->片段映射
        'y': y,
        'atom_mask': atoms_mask_batch
    }
```

---

## 4. 网络结构

### 4.1 FragNetLayerA（注意力层）

**文件**: `fragnet/model/gat/gat2.py`

FragNetLayerA 是核心注意力层，**同时处理四种图**。

```python
class FragNetLayerA(nn.Module):
    def __init__(self,
        atom_in=128, atom_out=128,
        frag_in=128, frag_out=128,
        edge_in=128, edge_out=128,
        fedge_in=128, num_heads=2,
        bond_edge_in=1, fbond_edge_in=8,
        return_attentions=False,
        bond_mask=None, frag_bond_mask=None, atom_mask_individual=None
    ):
        # 线性投影
        self.projection_a  = Linear(atom_in, (atom_out//num_heads)*num_heads)  # 原子投影
        self.projection_b  = Linear(edge_in, (edge_out//num_heads)*num_heads)  # 键投影
        self.projection_fb = Linear(fedge_in, (edge_out//num_heads)*num_heads) # 片段键投影

        # 注意力参数向量 (GAT风格)
        self.a     = Parameter(Tensor(num_heads, 2*atom_out//num_heads + edge_out))  # 原子注意力
        self.a_b   = Parameter(Tensor(num_heads, 2*edge_out//num_heads + edge_out)) # 键注意力
        self.f     = Parameter(Tensor(num_heads, 2*atom_out//num_heads + edge_out)) # 片段注意力
        self.f_a_b = Parameter(Tensor(num_heads, 2*edge_out//num_heads + edge_out)) # 片段键注意力

        # 可解释性的掩码参数
        self.bond_mask = bond_mask             # 掩码特定键
        self.frag_bond_mask = frag_bond_mask   # 掩码特定片段连接
        self.atom_mask_individual = atom_mask_individual  # 掩码特定原子
```

#### 前向传播逻辑（4步并行）

**Step 1: 键图注意力更新**（键节点特征更新）

```python
# 键图: 原子图中的边 -> 节点
target, source = edge_index_bonds_graph
node_feats_b = self.projection_b(node_feautures_bond_graph)
node_feats_b = node_feats_b.view(num_nodes_b, num_heads, -1)

source_features = node_feats_b[source]
target_features = node_feats_b[target]
ea_bonds = edge_attr_bond_graph.repeat(num_heads,1,1).permute(1,0,2)

# GAT式注意力: 拼接 target | edge | source，与学习向量 a_b 点积
message = torch.cat([target_features, ea_bonds, source_features], dim=-1)
attn_logits = torch.sum(message * self.a_b, dim=2)  # (n_edges, num_heads)
attn_logits = LeakyReLU(attn_logits)
attn_probs  = scatter_softmax(attn_logits, target, dim=0)  # 邻居归一化

# 加权聚合
node_feats_sum_b = scatter_add(attn_probs[...,None] * node_feats_b[source], target, dim=0)
new_bond_features = node_feats_sum_b.view(num_nodes_b, -1)

# 可解释性：可以掩码特定键
if self.bond_mask is not None:
    new_bond_features[self.bond_mask:self.bond_mask+2, :] = 0.0
```

**Step 2: 原子图注意力更新**（以更新后的键特征作为边特征）

```python
# 原子图：加入自环
edge_index, _ = add_self_loops(edge_index)
# 将新键特征拼接到边属性
edge_attr = torch.cat((new_bond_features, self_loop_attr), dim=0)

source, target = edge_index
node_features_atom_graph = self.projection_a(x_atoms)
node_features_atom_graph = node_features_atom_graph.view(num_nodes_a, num_heads, -1)

message = torch.cat([
    node_features_atom_graph[target],
    edge_attr.repeat(num_heads,1,1).permute(1,0,2),
    node_features_atom_graph[source]
], dim=-1)

attn_logits = torch.sum(message * self.a, dim=2)
attn_logits = LeakyReLU(attn_logits)
attn_probs  = scatter_softmax(attn_logits, target, dim=0)

node_feats_sum_a = scatter_add(attn_probs[...,None] * node_features_atom_graph[source], target, dim=0)
x_atoms_new = node_feats_sum_a.view(num_nodes_a, -1)

# 可解释性：可掩码特定原子
if self.atom_mask_individual is not None:
    x_atoms_new[self.atom_mask_individual, :] = 0.0

# 原子特征聚合到片段
x_frags = scatter_add(src=x_atoms_new, index=atom_to_frag_ids, dim=0)
```

**Step 3: 片段键图注意力更新**（片段间连接特征更新）

```python
# 片段键图: 片段连接 -> 节点
target, source = edge_index_fbond_graph
node_feats_fb = self.projection_fb(node_feautures_fbond_graph)

# 可解释性: 可掩码特定片段连接（双向边同时掩码）
if self.frag_bond_mask is not None:
    flipped_pairs = find_flipped_pairs(edge_attr_fbond_graph)
    edge_attr_fbond_graph[flipped_pairs_values[frag_bond_mask][0], :] = 0.0
    edge_attr_fbond_graph[flipped_pairs_values[frag_bond_mask][1], :] = 0.0

message = torch.cat([node_feats_fb[target], ea_fbonds, node_feats_fb[source]], dim=-1)
attn_logits = torch.sum(message * self.f_a_b, dim=2)
new_fbond_features = scatter_add(attn_probs[...,None]*node_feats_fb[source], target, dim=0)
new_fbond_features = new_fbond_features.view(num_nodes_fb, -1)
```

**Step 4: 片段图注意力更新**（以更新后的片段键特征作为边特征）

```python
source, target = frag_index
node_features_frag_graph = x_frags.view(num_nodes_f, num_heads, -1)

message = torch.cat([
    node_features_frag_graph[target],
    new_fbond_features.repeat(num_heads,1,1).permute(1,0,2),
    node_features_frag_graph[source]
], dim=-1)

attn_logits = torch.sum(message * self.f, dim=2)
attn_probs  = scatter_softmax(attn_logits, target, dim=0)
x_frags_new = scatter_add(attn_probs[...,None]*node_features_frag_graph[source], target, dim=0)
x_frags_new = x_frags_new.view(num_nodes_f, -1)

# 返回（可选返回注意力权重用于可视化）
if self.return_attentions:
    return (x_atoms_new, x_frags_new, new_bond_features, new_fbond_features,
            summed_attn_weights_atoms, summed_attn_weights_frags,
            summed_attn_weights_bonds, summed_attn_weights_fbonds)
else:
    return x_atoms_new, x_frags_new, new_bond_features, new_fbond_features
```

### 4.2 FragNet（骨干网络）

```python
class FragNet(nn.Module):
    def __init__(self, num_layer=4, drop_ratio=0.2, emb_dim=128,
                 atom_features=167, frag_features=167, edge_features=17,
                 fedge_in=6, fbond_edge_in=6, num_heads=4):

        self.layers = ModuleList()
        # 第一层: 输入维度 -> emb_dim
        self.layers.append(FragNetLayerA(
            atom_in=atom_features,  # 167
            atom_out=emb_dim,       # 128
            frag_in=frag_features,  # 167
            frag_out=emb_dim,       # 128
            edge_in=edge_features,  # 17
            edge_out=emb_dim,       # 128
            fedge_in=fedge_in,      # 6
            fbond_edge_in=fbond_edge_in,  # 6
            num_heads=num_heads     # 4
        ))
        # 后续层: emb_dim -> emb_dim
        for i in range(num_layer - 1):
            self.layers.append(FragNetLayerA(
                atom_in=emb_dim, atom_out=emb_dim,
                frag_in=emb_dim, frag_out=emb_dim,
                edge_in=emb_dim, edge_out=emb_dim,
                fedge_in=emb_dim, fbond_edge_in=fbond_edge_in,
                num_heads=num_heads
            ))

    def forward(self, batch):
        x_atoms = batch['x_atoms']     # (N_atoms, 167)
        x_frags = batch['x_frags']     # (N_frags, 167)

        # Dropout 应用于输入
        x_atoms = self.dropout(x_atoms)
        x_frags = self.dropout(x_frags)

        # 第一层
        x_atoms, x_frags, edge_features, fedge_features = self.layers[0](
            x_atoms, edge_index, edge_attr, frag_index, x_frags, atom_to_frag_ids,
            node_feautures_bond_graph, edge_index_bonds_graph, edge_attr_bond_graph,
            node_feautures_fbondg, edge_index_fbondg, edge_attr_fbondg
        )
        x_atoms = ReLU(Dropout(x_atoms))
        x_frags = ReLU(Dropout(x_frags))
        edge_features = ReLU(Dropout(edge_features))

        # 后续层（边特征从上一层传递）
        for layer in self.layers[1:]:
            x_atoms, x_frags, edge_features, fedge_features = layer(
                x_atoms, edge_index, edge_features,   # 注意: 使用上一层的 edge_features
                frag_index, x_frags, atom_to_frag_ids,
                edge_features, edge_index_bonds_graph, edge_attr_bond_graph,
                fedge_features, edge_index_fbondg, edge_attr_fbondg
            )
            x_atoms = ReLU(Dropout(x_atoms))
            x_frags = ReLU(Dropout(x_frags))

        return x_atoms, x_frags, edge_features, fedge_features
        # 输出维度: (N_atoms, 128), (N_frags, 128), (N_bonds, 128), (N_fbonds, 128)
```

### 4.3 回归/分类头

**FTHead3**（默认，4层 MLP）：

```python
class FTHead3(nn.Module):
    # 输入: 256维 (atom_pooled 128 + frag_pooled 128)
    # 架构: 256 -> h1(128) -> h2(1024) -> h3(1024) -> h4(512) -> n_classes
    def __init__(self, n_classes=1, input_dim=128,
                 h1=128, h2=1024, h3=1024, h4=512,
                 drop_ratio=0.1, act='relu'):
        self.layers = Sequential(
            Linear(input_dim*2, h1), Activation(),
            Dropout(drop_ratio),
            Linear(h1, h2), Activation(),
            Linear(h2, h3), Activation(),
            Linear(h3, h4), Activation(),
            Linear(h4, n_classes)
        )
```

其他可用头部：
- `FTHead1`: 2层 MLP (256 → 128 → 1)
- `FTHead2`: 3层 MLP (256 → [1024,1024,512] → 1)
- `FTHead4`: 2层 MLP，可配置激活函数

### 4.4 FragNetFineTune（下游任务模型）

```python
class FragNetFineTune(nn.Module):
    def __init__(self, n_classes=1, atom_features=167, frag_features=167,
                 edge_features=17, num_layer=4, num_heads=4, drop_ratio=0.15,
                 h1=256, h2=256, h3=256, h4=256, act='celu',
                 emb_dim=128, fthead='FTHead3'):

        self.pretrain = FragNet(...)    # 骨干网络
        self.fthead   = FTHead3(...)    # 回归头

    def forward(self, batch):
        # 1. 骨干前向传播
        x_atoms, x_frags, x_edge, x_fedge = self.pretrain(batch)

        # 2. 图级别池化 (sum pooling)
        x_atoms_pooled = scatter_add(x_atoms, batch['batch'], dim=0)       # (B, 128)
        x_frags_pooled = scatter_add(x_frags, batch['frag_batch'], dim=0)  # (B, 128)

        # 3. 拼接原子级和片段级表示
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), dim=1)           # (B, 256)

        # 4. 回归头输出
        x = self.fthead(cat)  # (B, n_classes)
        return x
```

---

## 5. 预训练

### 5.1 预训练任务与目标

FragNet 在 **3D 几何性质** 上进行自监督预训练，不需要任何标记数据：

| 任务 | 目标 | 来源 |
|------|------|------|
| 键长预测 | 每对成键原子间的距离 | MMFF优化的3D坐标 |
| 键角预测 | 每个原子的键角 | 3D坐标计算 |
| 二面角预测 | 键的扭转角 | 3D坐标计算 |
| 总能量预测 | 分子构象能量 | MMFF力场计算 |

### 5.2 PretrainTask 头部

```python
class PretrainTask(nn.Module):
    def __init__(self, dim_in=128, dim_out=1, L=2):
        # 键长头: 拼接两原子特征 + 边特征 -> 降维 -> 预测
        self.bl_reduce_layer = Linear(dim_in*3, dim_in)  # 384 -> 128
        self.bl_layers = [Linear(128,64), Linear(64,32), Linear(32,1)]

        # 键角头: 原子级特征 -> 预测（每原子一个角度）
        self.ba_layers = [Linear(128,64), Linear(64,32), Linear(32,1)]

        # 二面角头: 边特征 -> 预测（每条键一个角度）
        self.da_layers = [Linear(128,64), Linear(64,32), Linear(32,1)]

        # 能量头: 图级池化 -> 预测（每分子一个能量）
        self.FC_layers = [Linear(256,128), Linear(128,64), Linear(64,1)]

    def forward(self, x_atoms, x_frags, edge_attr, batch):
        # 键长: 拼接边两端原子特征 + 边特征
        bond_length_pred = concat(x_atoms[edge_index[0]], x_atoms[edge_index[1]], edge_attr)
        bond_length_pred = bl_reduce_layer(bond_length_pred)  # 384 -> 128
        bond_length_pred = bl_layers(bond_length_pred)  # -> 1 per bond

        # 键角: 直接从原子特征预测
        bond_angle_pred = ba_layers(x_atoms)  # -> 1 per atom

        # 二面角: 从边特征预测
        dihedral_angle_pred = da_layers(edge_attr)  # -> 1 per bond

        # 能量: 图级池化后预测
        x_atoms_pooled = scatter_add(x_atoms, batch['batch'], dim=0)
        x_frags_pooled = scatter_add(x_frags, batch['frag_batch'], dim=0)
        graph_rep = concat(x_atoms_pooled, x_frags_pooled)  # (B, 256)
        energy_pred = FC_layers(graph_rep)  # -> 1 per mol

        return bond_length_pred, bond_angle_pred, dihedral_angle_pred, energy_pred
```

### 5.3 训练流程

**文件**: `fragnet/train/pretrain/pretrain_gat2.py`

```python
# 启动命令
python train/pretrain/pretrain_gat2.py --config exps/pt/unimol_exp1s4/config.yaml

# 损失函数（四任务联合优化）
loss_fn = MSELoss()

for batch in train_loader:
    bond_length_pred, bond_angle_pred, dihedral_pred, energy_pred = model(batch)

    loss_bl  = loss_fn(bond_length_pred, bond_length_true)
    loss_ba  = loss_fn(bond_angle_pred,  bond_angle_true)
    loss_dih = loss_fn(dihedral_pred,    dihedral_true)
    loss_E   = loss_fn(energy_pred,      energy_true)

    total_loss = loss_bl + loss_ba + loss_dih + loss_E
    total_loss.backward()
    optimizer.step()
```

**超参数（默认值）：**
- `batch_size`: 512
- `lr`: 1e-4 (Adam)
- `n_epochs`: 200
- `es_patience`: 200 (早停耐心)
- 验证集比例: 10%

---

## 6. 微调（Fine-tuning）

### 6.1 配置系统

**文件**: `fragnet/exps/ft/esol/e1pt4.yaml`

```yaml
seed: 123
model_version: gat2

# 特征维度
atom_features: 167
frag_features: 167
edge_features: 17
fedge_in: 6
fbond_edge_in: 6

# 预训练模型位置
pretrain:
  num_layer: 4
  drop_ratio: 0.2
  num_heads: 4
  emb_dim: 128
  chkpoint_name: exps/pt/unimol_exp1s4/pt.pt

# 微调超参数
finetune:
  batch_size: 16
  lr: 1e-4
  model:
    n_classes: 1
    num_layer: 4
    drop_ratio: 0.1
    num_heads: 4
    emb_dim: 128
    h1: 128
    h2: 1024
    h3: 1024
    h4: 512
    act: relu
    fthead: FTHead3
  n_epochs: 10000
  target_type: regr   # regr | clsf
  loss: mse
  es_patience: 100    # 早停耐心
  train:
    path: finetune_data/moleculenet_exp1s/esol/train.pkl
  val:
    path: finetune_data/moleculenet_exp1s/esol/val.pkl
  test:
    path: finetune_data/moleculenet_exp1s/esol/test.pkl
```

### 6.2 训练循环

**文件**: `fragnet/train/finetune/finetune_gat2.py`

```python
python train/finetune/finetune_gat2.py --config exps/ft/esol/e1pt4.yaml
```

**核心流程：**
1. 加载预训练检查点
2. 计算训练集均值/标准差，归一化目标值
3. 训练循环（MSE/BCE损失 + Adam优化器）
4. 每 epoch 验证，早停监控验证集损失
5. 保存最优模型 checkpoint

```python
class Trainer:
    def train_regr(model, loader, optimizer, device):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch)
            loss = F.mse_loss(pred.squeeze(), batch['y'])
            loss.backward()
            optimizer.step()

    def validate_regr(model, loader, device):
        model.eval()
        with torch.no_grad():
            for batch in loader:
                pred = model(batch)
                # 计算 MSE

    def test(model, loader, device):
        # 返回 (score, true_labels, predictions)
```

### 6.3 超参数搜索

**文件**: `fragnet/hp/hpoptuna.py`

使用 Optuna 进行贝叶斯超参数优化：

```python
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_int('batch_size', 8, 128)
    drop_ratio = trial.suggest_float('drop_ratio', 0.05, 0.3)
    num_layer = trial.suggest_int('num_layer', 2, 6)
    act = trial.suggest_categorical('act', ['relu', 'gelu', 'silu', 'celu', 'selu'])

    model = FragNetFineTune(lr=lr, batch_size=batch_size, ...)
    val_loss = train_and_evaluate(model)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, callbacks=[pruner])
```

---

## 7. 可解释性（Interpretability）

FragNet 的可解释性是其核心创新，提供**四个粒度**的解释：

### 7.1 注意力权重提取

**文件**: `fragnet/vizualize/model.py`

通过设置 `return_attentions=True`，最后一层 FragNetLayerA 返回注意力权重：

```python
class FragNetViz(nn.Module):
    """在最后一层返回注意力权重"""
    def forward(self, batch):
        # 前几层正常前向
        for layer in self.layers[:-1]:
            x_atoms, x_frags, edge_features, fedge_features = layer(...)

        # 最后一层返回注意力权重
        last_layer = self.layers[-1]
        last_layer.return_attentions = True
        (x_atoms, x_frags, new_bond_features, new_fbond_features,
         attn_atoms,   # (N_atoms, num_heads)  原子注意力
         attn_frags,   # (N_frags, num_heads)  片段注意力
         attn_bonds,   # (N_bonds, num_heads)  键注意力
         attn_fbonds   # (N_fbonds, num_heads) 片段键注意力
        ) = last_layer(...)

        # 对多头注意力求和
        return (x_atoms, x_frags, edge_features, fedge_features,
                attn_atoms.sum(dim=1),   # (N_atoms,)
                attn_frags.sum(dim=1),   # (N_frags,)
                attn_bonds.sum(dim=1),   # (N_bonds,)
                attn_fbonds.sum(dim=1))  # (N_fbonds,)
```

### 7.2 掩码归因（Masking Attribution）

**文件**: `fragnet/vizualize/model_attr.py`

这是 FragNet 的**主要可解释性方法**：通过输入扰动计算每个结构单元对预测的贡献。

#### 核心思想

```
贡献度(element) = 预测值(不掩码) - 预测值(该element掩码为0)
```

正值 → 该元素增大预测值
负值 → 该元素减小预测值

#### 实现方式

```python
def get_predictions(dataset, model_config, chkpt_path, prop_type):
    # 1. 无掩码预测
    pred_no_mask = no_mask_predictions_property(dataset, chkpt_path, prop_type, args)

    # 2. 带掩码预测（dataset中每个样本是原分子对某个片段掩码后的版本）
    pred_mask = mask_predictions_property(dataset, chkpt_path, prop_type, args)

    # 3. 归因 = 原始预测 - 掩码后预测
    attribution = (pred_no_mask - pred_mask)
    return pred_no_mask, pred_mask, batch['y']
```

#### 数据准备（片段级掩码）

```python
def create_data(data, frag_type='brics'):
    """将每个分子复制 N_fragments 份，每份掩码一个片段"""
    copies = []
    for item in data:
        smiles = item.smiles
        graph, frags = get_frags(smiles, frag_type=frag_type)
        atoms_in_frags = get_atoms_in_frags(graph)

        for fid, frag_atoms in atoms_in_frags.items():
            data_copy = copy.deepcopy(item)
            atom_mask = torch.zeros(data_copy['x_atoms'].shape[0])
            atom_mask[frag_atoms] = 1  # 标记该片段的原子为掩码
            data_copy['atom_mask'] = atom_mask
            data_copy['frag_atoms'] = frag_atoms
            copies.append(data_copy)
    return copies
```

#### 模型中掩码的应用

```python
class FragNetFineTune(nn.Module):
    def forward(self, batch):
        x_atoms, x_frags, x_edge, x_fedge = self.pretrain(batch)

        if self.apply_mask:
            mask = batch['atom_mask']
            x_atoms[mask == 1] = 0.0  # 将被掩码片段的原子特征清零

        # 后续正常池化和预测...
```

#### 四种粒度的掩码

| 粒度 | 掩码方式 | 位置 |
|------|----------|------|
| 原子 | `x_atoms_new[atom_idx, :] = 0.0` | FragNetLayerA 内部 (`atom_mask_individual`) |
| 键 | `new_bond_features[bond_mask:bond_mask+2, :] = 0.0` | FragNetLayerA 内部 (`bond_mask`) |
| 片段 | `x_atoms[frag_atom_ids] = 0.0` | FragNetFineTune.forward() (`apply_mask`) |
| 片段连接 | `edge_attr_fbond_graph[pair_indices, :] = 0.0` | FragNetLayerA 内部 (`frag_bond_mask`) |

#### 归因权重计算与可视化

```python
def add_atom_weights(attribution, dataset):
    """将片段归因权重映射到每个原子"""
    atom_weights = {}
    for i in range(len(attribution)):
        w = attribution[i].item()           # 该片段的归因值
        frag_atoms = dataset[i]['frag_atoms']  # 该片段包含的原子
        for atom_id in frag_atoms:
            atom_weights[atom_id] = w       # 同一片段内原子权重相同
    return atom_weights

def highlight_atoms(atom_weights, mol):
    """用颜色热图渲染分子，颜色深浅表示贡献度"""
    cmap = cm.get_cmap('seismic_r', 10)       # 红蓝配色方案
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    highlightatoms = {}
    for atom_id, weight in atom_weights.items():
        highlightatoms[atom_id] = [cmap(norm(weight))]  # RGBA颜色

    d2d = rdMolDraw2D.MolDraw2DSVG(width=400, height=300)
    d2d.DrawMoleculeWithHighlights(mol, highlight_atom_map=highlightatoms, ...)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()  # SVG字符串
```

### 7.3 可视化系统

**文件**: `fragnet/vizualize/viz.py`

提供多种分子可视化功能：

```python
# 片段分解可视化
def highlight_frags(mol, frags, frag_type='brics'):
    """用不同颜色高亮显示每个片段"""
    colors = plt.cm.tab20c.colors  # 20色循环调色板
    for frag_id, frag_atoms in enumerate(frags):
        color = colors[frag_id % 20]
        for atom in frag_atoms:
            highlightatoms[atom] = [color]

# 注意力权重可视化
def highlight_frag_attention(mol, attn_weights, frags):
    """根据注意力权重着色"""
    cmap = cm.get_cmap('Reds')
    norm = Normalize(vmin=0, vmax=max(attn_weights))
    for frag_id, frag_atoms in enumerate(frags):
        color = cmap(norm(attn_weights[frag_id]))
        for atom in frag_atoms:
            highlightatoms[atom] = [color]
```

### 7.4 交互式 Web App

**文件**: `fragnet/vizualize/app.py`

基于 Streamlit 的交互式界面：

```bash
streamlit run fragnet/vizualize/app.py
```

**四个分析标签页：**

```
Tab 1: Atoms（原子级）
├── 分子结构图（原子按贡献度着色）
├── 每个原子的归因权重表格
└── 统计：均值/最大值/最小值

Tab 2: Bonds（键级）
├── 键贡献度热图
├── 每条键的权重排名表格
└── 键的起始/终止原子索引

Tab 3: Fragments（片段级）
├── 片段分解可视化（不同颜色）
├── 原子到片段的映射表格
├── 片段归因图（颜色深浅=重要性）
└── 片段贡献度排名表格

Tab 4: Fragment Connections（片段连接级）
├── 连接权重可视化
├── 片段间连接的重要性表格
└── 非共价相互作用分析
```

**输入方式：**
- 文本框输入 SMILES
- Ketcher 分子编辑器（交互式绘制）
- 下拉选择属性类型（溶解度/脂溶性/能量/DRP）

---

## 8. 整体数据流

```
SMILES 字符串
    │
    ▼
get_3Dcoords(smiles)
    │  RDKit MMFF94优化3D坐标
    │  (失败时退回2D坐标)
    ▼
FragmentedMol(mol, conf, frag_type='brics')
    │  BRICS.FindBRICSBonds() 找分解键
    │  remove_bond() 断开键
    │  Chem.GetMolFrags() 得到片段
    │  构建 Fragment 和 Connection 对象
    ▼
FeaturesEXP.get_atom_and_bond_features_atom_graph_one_hot()
    │  原子特征 (N_atoms, 167)
    │  键特征   (N_bonds*2, 17)
    │  片段特征 (N_frags, 167) = Σ原子特征
    │  连接特征 (N_connections, 6)
    ▼
CreateData.create_data_point()
    │  原子图: edge_index, edge_attr
    │  键图:   node_features_bonds, edge_index_bonds
    │  片段图: frag_index, edge_attr_fbonds
    │  片段键图: node_features_fbonds, edge_index_fbonds
    ▼ PyTorch Data 对象
collate_fn(data_list)
    │  批次拼接 + 节点索引偏移
    ▼ 批次字典
FragNet (骨干网络, 4层 FragNetLayerA)
    │  Layer 0: [167→128] 原子/片段/键/片段键 → 128维
    │  Layer 1-3: [128→128] 深化表示
    │  返回: x_atoms(N,128), x_frags(M,128), edge_feats(E,128), fedge_feats(F,128)
    ▼
FragNetFineTune.forward()
    │  scatter_add(x_atoms, batch) → (B, 128)  原子级图表示
    │  scatter_add(x_frags, frag_batch) → (B, 128)  片段级图表示
    │  cat → (B, 256)
    │  FTHead3: 256 → 128 → 1024 → 1024 → 512 → 1
    ▼ 预测值
    │
可解释性分析:
    ├── 注意力权重 (最后一层 return_attentions=True)
    │   ├── 原子注意力: (N_atoms,)
    │   ├── 键注意力:   (N_bonds,)
    │   ├── 片段注意力: (N_frags,)
    │   └── 片段键注意力: (N_fbonds,)
    │
    └── 掩码归因 (input perturbation)
        ├── 对每个片段掩码 → 重新预测
        │   attribution[i] = pred_nomask - pred_masked[i]
        ├── 同片段原子共享归因值
        └── 着色渲染 (seismic_r colormap: 蓝=负贡献, 红=正贡献)
```

---

## 9. 支持的任务与数据集

| 任务 | 模型类 | 数据集 |
|------|--------|--------|
| 分子性质回归 | `FragNetFineTune` | ESOL(logS), Lipophilicity, FreeSolv |
| 分子性质分类 | `FragNetFineTune` | BBBP, Tox21, SIDER |
| 分子能量预测 | `FragNetPreTrain` | 3D构象数据 |
| 药物-靶点相互作用 | `FragNetFineTuneDTA` | Davis, KIBA |
| 癌症药物反应预测 | `CDRPModel` | GDSC (分子特征+基因表达) |
| 多任务学习 | `FragNetFineTuneMultiTask` | 多性质联合预测 |

**CDRP（癌症药物反应预测）特殊处理：**
```python
class CDRPModel(nn.Module):
    def __init__(self, drug_model, gene_dim, device):
        self.drug_model = drug_model   # FragNetFineTuneBaseViz: 分子 -> 256维表示
        self.gene_encoder = Linear(gene_dim, 256)  # 基因表达 -> 256维
        self.head = Linear(512, 1)    # 拼接后预测

    def forward(self, batch):
        drug_repr = self.drug_model(batch)          # (B, 256)
        gene_repr = self.gene_encoder(batch['gene_expr'])  # (B, 256)
        combined = cat(drug_repr, gene_repr, dim=1)  # (B, 512)
        return self.head(combined)
```

---

## 10. 实验配置示例

### 预训练配置

```yaml
# exps/pt/unimol_exp1s4/config.yaml
seed: 123
exp_dir: exps/pt/unimol_exp1s4
model_version: gat2
atom_features: 167
frag_features: 167
edge_features: 17
fedge_in: 6
fbond_edge_in: 6

pretrain:
  num_layer: 4
  drop_ratio: 0.2
  num_heads: 4
  emb_dim: 128
  chkpoint_name: ${exp_dir}/pt.pt
  loss: mse
  batch_size: 512
  es_patience: 200
  lr: 1e-4
  n_epochs: 200
  data:
    - pretrain_data/esol/
```

### ESOL 溶解度微调配置

```yaml
# exps/ft/esol/e1pt4.yaml
seed: 123
model_version: gat2
atom_features: 167

pretrain:
  chkpoint_name: exps/pt/unimol_exp1s4/pt.pt
  num_layer: 4
  num_heads: 4
  emb_dim: 128

finetune:
  batch_size: 16
  lr: 1e-4
  model:
    fthead: FTHead3
    h1: 128
    h2: 1024
    h3: 1024
    h4: 512
    act: relu
    drop_ratio: 0.1
  n_epochs: 10000
  es_patience: 100
  target_type: regr
  loss: mse
  train:
    path: finetune_data/moleculenet_exp1s/esol/train.pkl
```

### 运行完整流程

```bash
# 1. 创建预训练数据
python data_create/create_pretrain_datasets.py

# 2. 创建微调数据
python data_create/create_finetune_datasets.py

# 3. 预训练
python train/pretrain/pretrain_gat2.py --config exps/pt/unimol_exp1s4/config.yaml

# 4. 微调 (ESOL溶解度)
python train/finetune/finetune_gat2.py --config exps/ft/esol/e1pt4.yaml

# 5. 启动交互式可视化App
streamlit run fragnet/vizualize/app.py

# 6. Jupyter Notebook 分析
jupyter notebook fragnet/notebooks/interprete.ipynb
```

---

## 关键设计总结

### 1. 多粒度信息传递
FragNet 的核心创新是**同时在四种图上进行消息传递**，且各图之间相互关联：
- 键图更新的特征作为原子图的边特征
- 原子特征聚合后更新片段特征
- 片段键图更新的特征作为片段图的边特征

### 2. 可解释性的双重路径
- **注意力权重**：直接从模型内部提取，反映模型在推理时对各结构单元的关注程度
- **掩码归因**：通过输入扰动，更直接地测量各结构单元对预测结果的因果影响

### 3. 预训练策略
通过3D几何性质（键长、键角、二面角、能量）进行自监督预训练，使模型学习到分子构象的物理先验知识，再迁移到下游性质预测任务。

### 4. 片段级表示
将分子分解为语义上有意义的片段（BRICS对应逆合成片段，Murcko对应药效团骨架），使得模型的解释与化学家的直觉相符。
