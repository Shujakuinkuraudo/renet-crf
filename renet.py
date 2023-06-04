import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ICEWS18
from torch_geometric.loader import DataLoader

from Myrenet import RENet, renet_crf
from test import BiLSTM_crf
import numpy as np
from tqdm import tqdm
from config import CFG
import wandb

if CFG.wandb:
    def wandb_init():
        config = {k: v for k, v in CFG.items() if '__' not in k}
        run = wandb.init(
            project=CFG.project,
            name=f"{CFG.model}-epoch-{CFG.epochs}",
            tags=CFG.tags,
            config=config,
            save_code=True
        )
        return run

# Load the dataset and precompute history objects.
path = osp.join("./", 'data', 'ICEWS18')
train_dataset = ICEWS18(path, pre_transform=RENet.pre_transform(CFG.seq_len))
test_dataset = ICEWS18(path, split='test')
train_loader = DataLoader(train_dataset[:int(CFG.train_percent * len(train_dataset))], batch_size=CFG.batch_size,
                          shuffle=True,
                          follow_batch=['h_sub', 'h_obj'])
test_loader = DataLoader(test_dataset[:int(CFG.test_percent * len(train_dataset))], batch_size=CFG.batch_size,
                         shuffle=False,
                         follow_batch=['h_sub', 'h_obj'])

# Initialize model and optimizer.
model = RENet(train_dataset.num_nodes, train_dataset.num_rels, hidden_channels=CFG.hidden_channels, seq_len=CFG.seq_len,
              dropout=0.2, num_tags=20).to(CFG.device)
if CFG.baseline_weight:
    checkpoint = torch.load(CFG.baseline_weight)
    model.load_state_dict(checkpoint)
model_crf_sub = BiLSTM_crf(CFG.hidden_channels, 100, 20, 5).to(CFG.device)
if CFG.crf_sub_weight:
    checkpoint = torch.load(CFG.crf_sub_weight)
    model_crf_sub.load_state_dict(checkpoint)
model_crf_obj = BiLSTM_crf(CFG.hidden_channels, 100, 20, 5).to(CFG.device)
if CFG.crf_obj_weight:
    checkpoint = torch.load(CFG.crf_obj_weight)
    model_crf_obj.load_state_dict(checkpoint)
optimizer = torch.optim.Adam(model.parameters(), 0.001)
optimizer_crf = torch.optim.Adam(
    [{'params': model_crf_sub.parameters(), "lr": 0.01}, {'params': model_crf_obj.parameters(), "lr": 0.01}])
if CFG.wandb:
    run = wandb_init()
    run.watch(model, log='all')
    Total_params = 0
    Trainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
    run.log({"Total params": Total_params, "Trainable params": Trainable_params})


def train(loader):
    model.train()
    # Train model via multi-class classification against the corresponding
    # object and subject entities.
    for data in tqdm(loader):
        data = data.to(CFG.device)
        optimizer.zero_grad()
        log_prob_obj, log_prob_sub = model(data)
        loss_obj = F.nll_loss(log_prob_obj, data.obj)
        loss_sub = F.nll_loss(log_prob_sub, data.sub)
        loss = loss_obj + loss_sub
        loss.backward()
        optimizer.step()


def train_crf(loader):
    model.eval()
    model_crf_obj.train()
    model_crf_sub.train()
    # Train model via multi-class classification against the corresponding
    # object and subject entities.
    for data in tqdm(loader):
        data = data.to(CFG.device)
        optimizer_crf.zero_grad()
        ent_sub, ent_obj, tags_sub, tags_obj = model.generate_crf_train(data)
        loss_1 = model_crf_sub(ent_sub, tags_sub)
        loss_2 = model_crf_obj(ent_obj, tags_obj)
        loss = loss_1 + loss_2
        if CFG.wandb:
            wandb.log({"loss": loss.item()})
        loss.backward()
        optimizer_crf.step()


def test(loader):
    model.eval()
    # Compute Mean Reciprocal Rank (MRR) and Hits@1/3/10.
    result = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float)
    for data in tqdm(loader):
        data = data.to(CFG.device)
        with torch.no_grad():
            log_prob_obj, log_prob_sub = model(data)
        result += model.test(log_prob_obj, data.obj) * data.obj.size(0)
        result += model.test(log_prob_sub, data.sub) * data.sub.size(0)
    result = result / (2 * len(loader.dataset))
    return result.tolist()  # %%


def test_crf(loader):
    model.eval()
    model_crf_obj.eval()
    model_crf_sub.eval()
    # Compute Mean Reciprocal Rank (MRR) and Hits@1/3/10.
    result = 0
    for data in tqdm(loader):
        data = data.to(CFG.device)
        with torch.no_grad():
            log_prob_obj, log_prob_sub = model(data)
        ent_sub, ent_obj = model.generate_crf_test(data)
        top_1_obj = torch.tensor(model_crf_sub.bilstm_forward(ent_sub)[:, -1]).to(CFG.device)
        top_1_sub = torch.tensor(model_crf_sub.bilstm_forward(ent_obj)[:, -1]).to(CFG.device)
        print(model_crf_sub.bilstm_forward(ent_sub), model_crf_sub.bilstm_forward(ent_obj))
        result += model.test_crf(log_prob_obj, data.obj, top_1_obj) * data.obj.size(0)
        result += model.test_crf(log_prob_sub, data.sub, top_1_sub) * data.sub.size(0)
    result = result / (2 * len(loader.dataset))
    print(result)
    return result  # %%


if CFG.baseline_train:
    for epoch in range(CFG.epochs):
        train(train_loader)
        mrr, hits1, hits2, hits3, hits10 = test(test_loader)
        if CFG.wandb:
            wandb.log({"epoch": epoch, "MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10})
        torch.save(model.state_dict(), "weights/" + str(epoch) + ".pt")

model.cluster(5)
for epoch_crf in range(CFG.epochs):
    if CFG.crf_train:
        train_crf(train_loader)
        torch.save(model_crf_sub.state_dict(), "weights/crf_20_20_sub_" + str(epoch_crf) + ".pt")
        torch.save(model_crf_obj.state_dict(), "weights/crf_20_20_obj_" + str(epoch_crf) + ".pt")

    hits1 = test_crf(test_loader)
    if CFG.wandb:
        wandb.log({"epoch": epoch_crf, "Hits@1": hits1})

if CFG.wandb:
    wandb.log({"epoch": epoch, "MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10})
if CFG.wandb:
    run.finish()
