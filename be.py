from easydict import EasyDict
from opt import *
import itertools


def beale_function(x: torch.tensor):
    return (1.5 - x[1] + x[1] * x[2]) ** 2 + (2.25 - x[1] + x[1] * x[2] ** 2) ** 2 + (
            2.625 - x[1] + x[1] * x[2] ** 3) ** 2


for start_point, model in itertools.product(
        [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 1), (0, 3, 1), (0, 3, 3), (0, 3, 5), (0, 3, 2), (0, 3, 4),
         (0, 4, 4)],
        ["GD", "Adam"]):
    CFG = EasyDict()
    CFG.start_point = start_point
    CFG.model = model
    CFG.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG.project = "opt"
    CFG.lr = 1e-3
    CFG.wandb = True
    CFG.max_iter = int(3 * 1e5)
    CFG.tags = ["v3"]
    if CFG.wandb:
        # from kaggle_secrets import UserSecretsClient
        #
        # user_secrets = UserSecretsClient()
        # secret_value_0 = user_secrets.get_secret("WANDB")
        # wandb.login(key=secret_value_0)
        def wandb_init():
            run = wandb.init(
                project=CFG.project,
                name=f"{CFG.project}-{CFG.model}-{CFG.start_point}",
                config={k: v for k, v in CFG.items() if '__' not in k},
                tags=CFG.tags,
                save_code=True
            )
            return run
    if CFG.wandb:
        run = wandb_init()
    g = eval(CFG.model)(CFG, beale_function, max_iter=CFG.max_iter, lr=CFG.lr)
    g(torch.Tensor(CFG.start_point))
    if CFG.wandb:
        run.finish()
