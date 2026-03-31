from fragnet.model.gcn.gcn import FragNetPreTrain
from fragnet.dataset.dataset import load_data_parts
from fragnet.dataset.data import mask_atom_features
import torch.nn as nn
from fragnet.train.utils import EarlyStopping
import torch
from fragnet.dataset.data import collate_fn
from torch.utils.data import DataLoader
from fragnet.dataset.features import atom_list_one_hot
from tqdm import tqdm
import pickle
import os
from sklearn.model_selection import train_test_split
import argparse
from omegaconf import OmegaConf


def load_ids(fn, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train_ids.pkl')
    val_path   = os.path.join(output_dir, 'val_ids.pkl')
    test_path  = os.path.join(output_dir, 'test_ids.pkl')

    if not os.path.exists(train_path):
        train_ids, test_ids = train_test_split(fn, test_size=0.2, random_state=42)
        test_ids, val_ids   = train_test_split(test_ids, test_size=0.5, random_state=42)

        with open(train_path, 'wb') as f:
            pickle.dump(train_ids, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_ids, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_ids, f)
    else:
        with open(train_path, 'rb') as f:
            train_ids = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_ids = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_ids = pickle.load(f)

    return train_ids, val_ids, test_ids


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # ✅ 先取 label，再 mask
        labels = batch['x_atoms'][:, :len(atom_list_one_hot)].argmax(1)
        mask_atom_features(batch)

        for k, v in batch.items():
            batch[k] = batch[k].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(loader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            # ✅ 先取 label，再 mask
            labels = batch['x_atoms'][:, :len(atom_list_one_hot)].argmax(1)
            mask_atom_features(batch)

            for k, v in batch.items():
                batch[k] = batch[k].to(device)
            labels = labels.to(device)

            out = model(batch)
            loss = loss_fn(out, labels)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yaml", type=str, required=False, default='config.yaml')
    args = parser.parse_args()

    if args.config:
        opt = OmegaConf.load(args.config)
        OmegaConf.resolve(opt)
        args = opt

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # 加载全部数据，在数据条目层面做 split
    all_data = load_data_parts(args.data_dir)


    print(f"all_data 长度: {len(all_data)}")
    print(f"all_data[0] 类型: {type(all_data[0])}")
    print(f"all_data[0]: {all_data[0]}")

    fn = list(range(len(all_data)))
    
    train_ids, val_ids, test_ids = load_ids(fn, args.output_dir)

    train_dataset = [all_data[i] for i in train_ids]
    val_dataset   = [all_data[i] for i in val_ids]

    print(f"total: {len(all_data)}, train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, collate_fn=collate_fn,
                              batch_size=args.val_batch_size, shuffle=False, drop_last=False)

    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    model_pretrain = FragNetPreTrain(
        atom_features=args.atom_features,
        frag_features=args.frag_features,
        edge_features=args.edge_features,
    )
    model_pretrain.to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_pretrain.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                   chkpoint_name=args.chkpoint_name)

    for epoch in tqdm(range(args.epochs)):

        train_loss = train(model_pretrain, train_loader, optimizer, device)
        val_loss   = validate(val_loader, model_pretrain, device)
        print(train_loss, val_loss)

        early_stopping(val_loss, model_pretrain)

        if early_stopping.early_stop:
            print("Early stopping")
            break