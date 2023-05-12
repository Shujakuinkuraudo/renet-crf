import torch
import numpy as np
import wandb
from tqdm import tqdm


class Optimizer:
    def __init__(self, CFG, function, max_iter):
        self.CFG = CFG
        self.function = function
        self.max_iter = max_iter

    @staticmethod
    def grad(f, x: torch.tensor):
        if not x.requires_grad: x = x.requires_grad_()
        return torch.autograd.grad(outputs=f(x), inputs=x)[0].detach()

    def forward(self, x):
        return self.function(x).detach().item()

    def point(self, x: torch.tensor):
        p = x.tolist()
        p.append(self.forward(x))
        return p

    def plot(self, points, x_range, y_range):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        points = torch.tensor(points).view(-1, 1, 3)
        # 创建 Line3DCollection 对象并添加到 Axes3D 中
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line = Line3DCollection(segments, cmap=plt.get_cmap('jet'), linewidth=3)
        line.set_array(np.linspace(0, 1, len(points)))
        ax.add_collection(line)

        # 定义函数
        def func(x, y):
            return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

        # 生成 x 和 y 坐标数据
        X, Y = np.meshgrid(np.linspace(*x_range, 100), np.linspace(*y_range, 100))

        # 绘制 3D 图像,
        ax.plot_surface(X, Y, func(X, Y), cmap='BrBG', alpha=0.5)

        ax.scatter(3, 0.5, 0, marker='*', s=300, alpha=0.3)

        # 调整坐标轴和显示图形
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        if self.CFG.wandb:
            wandb.log({"fig": wandb.Image(fig)})

    def cal(self, x):
        pass

    def __call__(self, x, plot=True):
        x_, points = self.cal(x)
        if plot:
            x = x.detach().numpy()
            self.plot(points, x_range=[min(x[1], -2), max(3.2, x[1])], y_range=[min(x[2], -2), max(x[2], 3.2)])
        if self.CFG.wandb:
            wandb.log({"end_point": x_})
        return x_

    def log(self, epoch, function_value, delta_value):
        if self.CFG.wandb:
            if epoch % (self.max_iter // 1000) == 0:
                wandb.log({"epoch": epoch, "function_value": function_value, "delta_value": delta_value})
        else:
            if epoch % (self.max_iter // 10) == 0:
                tqdm.write(str(self.forward(function_value)), end="")


class GD(Optimizer):
    def __init__(self, CFG, function, lr=1e-2, iter_eps=1e-6, max_iter=1000):
        super().__init__(CFG, function, max_iter)
        self.lr = lr
        self.iter_eps = iter_eps

    def cal(self, x):

        import time
        T1 = time.time_ns()

        points = []
        for _ in tqdm(range(self.max_iter)):
            g_t = self.grad(self.function, x)

            delta = self.lr * g_t
            if torch.norm(delta) <= self.iter_eps:
                if self.CFG.wandb:
                    wandb.log({"finish_epoch": _})
                break
            x = x - delta
            points.append(self.point(x)[1:])
            self.log(_, self.forward(x), torch.norm(delta))

        T2 = time.time_ns()
        if self.CFG.wandb:
            wandb.log({"time_ms": (T2 - T1) / 1e9})
        return x, points


class Adam(Optimizer):
    def __init__(self, CFG, function, lr=1e-2, iter_eps=1e-6, max_iter=1000, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__(CFG, function, max_iter)
        self.lr = lr
        self.iter_eps = iter_eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

    def cal(self, x):
        import time
        T1 = time.time_ns()

        points = []
        for _ in tqdm(range(self.max_iter)):
            g_t = self.grad(self.function, x)
            if _ == 0:
                m_t = torch.zeros(x.size())
                v_t = torch.zeros(x.size())
            m_t = self.beta_1 * m_t + (1 - self.beta_1) * g_t
            v_t = self.beta_2 * v_t + (1 - self.beta_2) * (g_t ** 2)

            delta = self.lr * (m_t / (1 - self.beta_1)) / (torch.sqrt(v_t / (1 - self.beta_2)) + self.eps)
            if torch.norm(delta) <= self.iter_eps:
                wandb.log({"finish_epoch": _})
                break
            x = x - delta
            points.append(self.point(x)[1:])
            self.log(_, self.forward(x), torch.norm(delta))
        T2 = time.time_ns()
        if self.CFG.wandb:
            wandb.log({"time_ms": (T2 - T1) / 1e9})
        return x, points
