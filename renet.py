import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ICEWS18
from torch_geometric.loader import DataLoader
from Myrenet import RENet
from config import CFG
import numpy as np
from tqdm import tqdm

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
train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, follow_batch=['h_sub', 'h_obj'])
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, follow_batch=['h_sub', 'h_obj'])

# Initialize model and optimizer.
model = RENet(train_dataset.num_nodes, train_dataset.num_rels, hidden_channels=CFG.hidden_channels, seq_len=CFG.seq_len,
              dropout=0.2, ).to(CFG.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
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


for epoch in range(CFG.epochs):
    train(train_loader)
    print(model.cluster(20))
    mrr, hits1, hits3, hits10 = test(test_loader)
    if CFG.wandb:
        wandb.log({"epoch": epoch, "MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10})
torch.save(model.state_dict(), "weights/a.pt")
if CFG.wandb:
    run.finish()
