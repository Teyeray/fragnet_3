import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pickle
import os
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

from fragnet.model.gcn.gcn import FragNetFineTune
from fragnet.dataset.dataset import load_pickle_dataset
from fragnet.dataset.data import collate_fn
from fragnet.train.utils import EarlyStopping, TrainerFineTune as Trainer


# ─────────────────────────── 工具函数 ───────────────────────────

def seed_everything(seed: int):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_train_stats(ds, exp_dir):
    """计算训练集 y 的均值和标准差，并保存"""
    y_old = np.array([i.y.item() for i in ds])
    mean = np.mean(y_old)
    sdev = np.std(y_old)
    os.makedirs(exp_dir, exist_ok=True)
    with open(f'{exp_dir}/train_stats.pkl', 'wb') as f:
        pickle.dump({'mean': mean, 'sdev': sdev}, f)
    return mean, sdev


def scale_y(ds, mean, sdev):
    """对数据集的 y 做 z-score 标准化"""
    y_old = np.array([i.y.item() for i in ds])
    y = (y_old - mean) / sdev
    for i, d in enumerate(ds):
        d.y = torch.tensor([y[i]], dtype=torch.float)


def save_predictions(trainer, loader, model, exp_dir, device,
                      save_name='test_res', loss_type='mse', seed=123):
    """保存预测结果到 pkl"""
    score, true, pred = trainer.test(model=model, loader=loader, device=device)
    smiles = [i.smiles for i in loader.dataset]

    if loss_type == 'mse':
        print(f'{save_name} rmse: ', score ** 0.5)
        res = {'acc': score ** 0.5, 'true': true, 'pred': pred, 'smiles': smiles}
    elif loss_type in ('cel', 'bce'):
        print(f'{save_name} auc: ', score)
        res = {'acc': score, 'true': true, 'pred': pred, 'smiles': smiles}

    with open(f"{exp_dir}/{save_name}_{seed}.pkl", 'wb') as f:
        pickle.dump(res, f)


# ─────────────────────────── 主流程 ───────────────────────────

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yaml",
                        type=str, required=False, default='config.yaml')
    args = parser.parse_args()

    if args.config:
        opt = OmegaConf.load(args.config)
        OmegaConf.resolve(opt)
        opt.update(vars(args))
        args = opt

    seed_everything(args.seed)

    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    writer = SummaryWriter(exp_dir + '/runs')

    # ── 构建 GCN FineTune 模型 ──
    model = FragNetFineTune(
        n_classes=args.finetune.model.n_classes,
        atom_features=args.atom_features,
        frag_features=args.frag_features,
        edge_features=args.edge_features,
        num_layer=args.finetune.model.num_layer,
        drop_ratio=args.finetune.model.drop_ratio,
        emb_dim=args.finetune.model.emb_dim,
    )

    trainer = Trainer(target_type=args.finetune.target_type)

    # ── 加载预训练权重（可选）──
    pt_chkpoint_name = args.pretrain.chkpoint_name
    if pt_chkpoint_name:
        from fragnet.model.gcn.gcn import FragNetPreTrain

        # ✅ 必须用预训练时的结构参数重建模型，而非 finetune 的参数
        modelpt = FragNetPreTrain(
            atom_features=args.atom_features,
            frag_features=args.frag_features,
            edge_features=args.edge_features,
            emb_dim=args.pretrain.emb_dim,        # ✅ 预训练时的 emb_dim（如 128）
            num_layer=args.pretrain.num_layer,     # ✅ 预训练时的 num_layer（如 6）
            drop_ratio=args.pretrain.drop_ratio,   # ✅ 预训练时的 drop_ratio
        )
        modelpt.load_state_dict(
            torch.load(pt_chkpoint_name, map_location=device, weights_only=False)
        )

        # ✅ strict=False：finetune 与 pretrain 结构不同时，只加载匹配的层
        missing, unexpected = model.pretrain.load_state_dict(
            modelpt.pretrain.state_dict(), strict=False
        )
        print(f'预训练权重已加载: {pt_chkpoint_name}')
        if missing:
            print(f'  [warn] missing keys ({len(missing)}): {missing}')
        if unexpected:
            print(f'  [warn] unexpected keys ({len(unexpected)}): {unexpected}')
    else:
        print('未使用预训练权重')

    # ── 加载数据集 ──
    train_dataset = load_pickle_dataset(args.finetune.train.path)
    val_dataset   = load_pickle_dataset(args.finetune.val.path)
    test_dataset  = load_pickle_dataset(args.finetune.test.path)

    # 可选：对回归任务做 y 标准化
    if args.finetune.get('scale_y', False):
        mean, sdev = get_train_stats(train_dataset, exp_dir)
        scale_y(train_dataset, mean, sdev)
        scale_y(val_dataset,   mean, sdev)
        scale_y(test_dataset,  mean, sdev)
        print(f'y 标准化: mean={mean:.4f}, sdev={sdev:.4f}')

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn,
                              batch_size=args.finetune.batch_size,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset, collate_fn=collate_fn,
                              batch_size=64, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset, collate_fn=collate_fn,
                              batch_size=64, shuffle=False, drop_last=False)

    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # ── 损失函数 / 优化器 / 调度器 ──
    if args.finetune.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.finetune.loss == 'cel':
        loss_fn = nn.CrossEntropyLoss()
    elif args.finetune.loss == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = None

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune.lr)

    scheduler = None
    if args.finetune.get('use_schedular', False):
        scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=30
        )

    ft_chk_point = args.finetune.chkpoint_name
    early_stopping = EarlyStopping(
        patience=args.finetune.es_patience,
        verbose=True,
        chkpoint_name=ft_chk_point,
    )

    # ── 训练循环 ──
    for epoch in tqdm(range(args.finetune.n_epochs)):

        train_loss = trainer.train(
            model=model, loader=train_loader,
            optimizer=optimizer, scheduler=scheduler,
            device=device, val_loader=val_loader,
        )
        val_loss, _, _ = trainer.test(model=model, loader=val_loader, device=device)

        print(f"epoch {epoch:>4d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # ── 加载最优权重并保存预测结果 ──
    model.load_state_dict(torch.load(ft_chk_point, map_location=device, weights_only=False))

    save_predictions(trainer=trainer, loader=val_loader,  model=model,
                     exp_dir=exp_dir, device=device,
                     save_name='val_res',  loss_type=args.finetune.loss, seed=args.seed)
    save_predictions(trainer=trainer, loader=test_loader, model=model,
                     exp_dir=exp_dir, device=device,
                     save_name='test_res', loss_type=args.finetune.loss, seed=args.seed)

    writer.close()
    print("Done.")