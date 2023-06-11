from easydict import EasyDict
import torch

CFG = EasyDict()
CFG.project = "temporal-knowledge-base-completion"
CFG.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG.epochs = 21
CFG.seq_len = 5
CFG.model = "Renet"
CFG.tags = "Baseline-1"
CFG.batch_size = 4096
CFG.hidden_channels = 100
CFG.wandb = True
CFG.baseline_weight = None
CFG.crf_obj_weight = None
CFG.crf_sub_weight = None
CFG.crf_train = True
CFG.baseline_train = True
CFG.train_percent = 0.5
CFG.test_percent = 0.5
