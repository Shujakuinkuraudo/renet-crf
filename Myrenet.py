import math
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRU, Linear, Parameter
from torch_scatter import scatter_mean

from torch_geometric.data.data import Data
from test import BiLSTM_crf
from config import CFG


class RENet(torch.nn.Module):
    r"""The Recurrent Event Network model from the `"Recurrent Event Network
    for Reasoning over Temporal Knowledge Graphs"
    <https://arxiv.org/abs/1904.05530>`_ paper

    .. math::
        f_{\mathbf{\Theta}}(\mathbf{e}_s, \mathbf{e}_r,
        \mathbf{h}^{(t-1)}(s, r))

    based on a RNN encoder

    .. math::
        \mathbf{h}^{(t)}(s, r) = \textrm{RNN}(\mathbf{e}_s, \mathbf{e}_r,
        g(\mathcal{O}^{(t)}_r(s)), \mathbf{h}^{(t-1)}(s, r))

    where :math:`\mathbf{e}_s` and :math:`\mathbf{e}_r` denote entity and
    relation embeddings, and :math:`\mathcal{O}^{(t)}_r(s)` represents the set
    of objects interacted with subject :math:`s` under relation :math:`r` at
    timestamp :math:`t`.
    This model implements :math:`g` as the **Mean Aggregator** and
    :math:`f_{\mathbf{\Theta}}` as a linear projection.

    Args:
        num_nodes (int): The number of nodes in the knowledge graph.
        num_rels (int): The number of relations in the knowledge graph.
        hidden_channels (int): Hidden size of node and relation embeddings.
        seq_len (int): The sequence length of past events.
        num_layers (int, optional): The number of recurrent layers.
            (default: :obj:`1`)
        dropout (float): If non-zero, introduces a dropout layer before the
            final prediction. (default: :obj:`0.`)
        bias (bool, optional): If set to :obj:`False`, all layers will not
            learn an additive bias. (default: :obj:`True`)
    """

    def __init__(
            self,
            num_nodes: int,
            num_rels: int,
            hidden_channels: int,
            seq_len: int,
            num_layers: int = 1,
            dropout: float = 0.,
            bias: bool = True,
            num_tags: int = 20
    ):
        super().__init__()

        self.cluster_labels = None
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = dropout
        self.num_tags = num_tags

        self.ent = Parameter(torch.Tensor(num_nodes, hidden_channels))
        self.rel = Parameter(torch.Tensor(num_rels, hidden_channels))

        self.sub_gru = GRU(3 * hidden_channels, hidden_channels, num_layers,
                           batch_first=True, bias=bias)
        self.obj_gru = GRU(3 * hidden_channels, hidden_channels, num_layers,
                           batch_first=True, bias=bias)

        self.sub_lin = Linear(3 * hidden_channels, num_nodes, bias=bias)
        self.obj_lin = Linear(3 * hidden_channels, num_nodes, bias=bias)

        self.lstm_crf_sub = BiLSTM_crf(self.hidden_channels, 5, 5, self.num_tags)
        self.lstm_crf_obj = BiLSTM_crf(self.hidden_channels, 5, 5, self.num_tags)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.ent, gain=math.sqrt(2.0))
        torch.nn.init.xavier_uniform_(self.rel, gain=math.sqrt(2.0))

        self.sub_gru.reset_parameters()
        self.obj_gru.reset_parameters()
        self.sub_lin.reset_parameters()
        self.obj_lin.reset_parameters()

    @staticmethod
    def pre_transform(seq_len: int) -> Callable:
        r"""Precomputes history objects

        .. math::
            \{ \mathcal{O}^{(t-k-1)}_r(s), \ldots, \mathcal{O}^{(t-1)}_r(s) \}

        of a :class:`torch_geometric.datasets.icews.EventDataset` with
        :math:`k` denoting the sequence length :obj:`seq_len`.
        """

        class PreTransform(object):
            def __init__(self, seq_len: int):
                self.seq_len = seq_len
                self.inc = 5000
                self.t_last = 0
                self.sub_hist = self.increase_hist_node_size([])
                self.obj_hist = self.increase_hist_node_size([])

            def increase_hist_node_size(self, hist: List[int]) -> List[int]:
                hist_inc = torch.zeros((self.inc, self.seq_len + 1, 0))
                return hist + hist_inc.tolist()

            def get_history(
                    self,
                    hist: List[int],
                    node: int,
                    rel: int,
            ) -> Tuple[Tensor, Tensor]:
                hists, ts = [], []
                for s in range(seq_len):
                    h = hist[node][s]
                    hists += h
                    ts.append(torch.full((len(h),), s, dtype=torch.long))
                node, r = torch.tensor(hists, dtype=torch.long).view(-1, 2).t().contiguous()
                node = node[r == rel]
                t = torch.cat(ts, dim=0)[r == rel]
                return node, t

            def step(self, hist: List[int]) -> List[int]:
                for i in range(len(hist)):
                    hist[i] = hist[i][1:]
                    hist[i].append([])
                return hist

            def __call__(self, data: Data) -> Data:
                sub, rel, obj, t = data.sub, data.rel, data.obj, data.t
                if max(sub, obj) + 1 > len(self.sub_hist):  # pragma: no cover
                    self.sub_hist = self.increase_hist_node_size(self.sub_hist)
                    self.obj_hist = self.increase_hist_node_size(self.obj_hist)

                # Delete last timestamp in history.
                if t > self.t_last:
                    self.sub_hist = self.step(self.sub_hist)
                    self.obj_hist = self.step(self.obj_hist)
                    self.t_last = t

                # Save history in data object.
                data.h_sub, data.h_sub_t = self.get_history(
                    self.sub_hist, sub, rel)
                data.h_obj, data.h_obj_t = self.get_history(
                    self.obj_hist, obj, rel)

                # Add new event to history.
                self.sub_hist[sub][-1].append([obj, rel])
                self.obj_hist[obj][-1].append([sub, rel])

                return data

            def __repr__(self) -> str:  # pragma: no cover
                return f'{self.__class__.__name__}(seq_len={self.seq_len})'

        return PreTransform(seq_len)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """Given a :obj:`data` batch, computes the forward pass.

        Args:
            data (torch_geometric.data.Data): The input data, holding subject
                :obj:`sub`, relation :obj:`rel` and object :obj:`obj`
                information with shape :obj:`[batch_size]`.
                In addition, :obj:`data` needs to hold history information for
                subjects, given by a vector of node indices :obj:`h_sub` and
                their relative timestamps :obj:`h_sub_t` and batch assignments
                :obj:`h_sub_batch`.
                The same information must be given for objects (:obj:`h_obj`,
                :obj:`h_obj_t`, :obj:`h_obj_batch`).
        """

        assert 'h_sub_batch' in data and 'h_obj_batch' in data
        batch_size, seq_len = data.sub.size(0), self.seq_len
        h_sub_t = data.h_sub_t + data.h_sub_batch * seq_len
        h_obj_t = data.h_obj_t + data.h_obj_batch * seq_len
        h_sub = scatter_mean(self.ent[data.h_sub], h_sub_t, dim=0,
                             dim_size=batch_size * seq_len).view(
            batch_size, seq_len, -1)

        h_obj = scatter_mean(self.ent[data.h_obj], h_obj_t, dim=0,
                             dim_size=batch_size * seq_len).view(
            batch_size, seq_len, -1)

        ent_sub_class = torch.zeros(self.ent.size(0), 5, self.ent.size(1)).cuda()
        ent_obj_class = torch.zeros(self.ent.size(0), 5, self.ent.size(1)).cuda()
        ent_sub_class[data.h_sub, data.h_sub_t] = self.ent[data.h_sub]
        ent_obj_class[data.h_obj, data.h_obj_t] = self.ent[data.h_obj]

        sub = self.ent[data.sub].unsqueeze(1).repeat(1, seq_len, 1)
        rel = self.rel[data.rel].unsqueeze(1).repeat(1, seq_len, 1)
        obj = self.ent[data.obj].unsqueeze(1).repeat(1, seq_len, 1)

        _, h_sub = self.sub_gru(torch.cat([sub, h_sub, rel], dim=-1))
        _, h_obj = self.obj_gru(torch.cat([obj, h_obj, rel], dim=-1))
        h_sub, h_obj = h_sub.squeeze(0), h_obj.squeeze(0)

        h_sub = torch.cat([self.ent[data.sub], h_sub, self.rel[data.rel]],
                          dim=-1)
        h_obj = torch.cat([self.ent[data.obj], h_obj, self.rel[data.rel]],
                          dim=-1)

        h_sub = F.dropout(h_sub, p=self.dropout, training=self.training)
        h_obj = F.dropout(h_obj, p=self.dropout, training=self.training)

        log_prob_obj = F.log_softmax(self.sub_lin(h_sub), dim=1)
        log_prob_sub = F.log_softmax(self.obj_lin(h_obj), dim=1)

        return log_prob_obj, log_prob_sub

    def cluster(self, num_tags) -> Tensor:
        from sklearn.cluster import KMeans
        # 将张量转换为NumPy数组
        ent_np = self.ent.detach().cpu().numpy()

        # 使用K-means进行聚类
        kmeans = KMeans(n_clusters=num_tags, n_init=10)
        labels = kmeans.fit_predict(ent_np)

        # 将聚类标签转换为PyTorch张量
        cluster_labels = torch.from_numpy(labels)
        self.cluster_labels = cluster_labels.long().to(CFG.device)

    def generate_crf_train(self, data: Data) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        seq_len_mul_hidden = self.seq_len * self.hidden_channels
        ent_sub = torch.zeros(data.sub.size(0), seq_len_mul_hidden).to(CFG.device)
        ent_obj = torch.zeros(data.sub.size(0), seq_len_mul_hidden).to(CFG.device)
        index_sub = [seq_len_mul_hidden - self.hidden_channels] * data.sub.size(0)
        index_obj = [seq_len_mul_hidden - self.hidden_channels] * data.obj.size(0)

        tags_sub = torch.full((data.sub.size(0), self.seq_len), -1, dtype=torch.long).to(CFG.device)
        tags_obj = torch.full((data.sub.size(0), self.seq_len), -1, dtype=torch.long).to(CFG.device)

        index_tags_sub = [self.seq_len - 2] * data.sub.size(0)
        index_tags_obj = [self.seq_len - 2] * data.obj.size(0)

        tags_sub[:, -1] = self.cluster_labels[data.obj[:]]

        tags_obj[:, -1] = self.cluster_labels[data.sub[:]]

        def fill_ent_tags(ent, index_ent, tags, index_tags, h_batch, h, hidden_channels):
            for i in range(h.size(0) - 1, -1, -1):
                if index_ent[h_batch[i]] >= 0:
                    ent[h_batch[i], index_ent[h_batch[i]]:index_ent[h_batch[i]] + hidden_channels] = self.ent[
                        h_batch[i]]
                if index_tags[h_batch[i]] >= 0:
                    tags[h_batch[i], index_tags[h_batch[i]]] = self.cluster_labels[h[i]]
                index_ent[h_batch[i]] -= hidden_channels
                index_tags[h_batch[i]] -= 1

        fill_ent_tags(ent_sub, index_sub, tags_sub, index_tags_sub, data.h_sub_batch, data.h_sub, self.hidden_channels)
        fill_ent_tags(ent_obj, index_obj, tags_obj, index_tags_obj, data.h_obj_batch, data.h_obj, self.hidden_channels)

        ent_sub = ent_sub.view(data.sub.size(0), self.seq_len, self.hidden_channels)
        ent_obj = ent_obj.view(data.sub.size(0), self.seq_len, self.hidden_channels)

        return ent_sub, ent_obj, tags_sub, tags_obj

    def generate_crf_test(self, data: Data) -> Tuple[Tensor, Tensor]:
        seq_len_mul_hidden = self.seq_len * self.hidden_channels
        ent_sub = torch.zeros(data.sub.size(0), seq_len_mul_hidden).to(CFG.device)
        ent_obj = torch.zeros(data.sub.size(0), seq_len_mul_hidden).to(CFG.device)
        index_sub = [seq_len_mul_hidden - self.hidden_channels] * data.sub.size(0)
        index_obj = [seq_len_mul_hidden - self.hidden_channels] * data.obj.size(0)

        def fill_ent_tags(ent, index_ent, h_batch, h, hidden_channels):
            for i in range(h.size(0) - 1, -1, -1):
                if index_ent[h_batch[i]] >= 0:
                    ent[h_batch[i], index_ent[h_batch[i]]:index_ent[h_batch[i]] + hidden_channels] = self.ent[
                        h_batch[i]]
                index_ent[h_batch[i]] -= hidden_channels

        fill_ent_tags(ent_sub, index_sub, data.h_sub_batch, data.h_sub, self.hidden_channels)
        fill_ent_tags(ent_obj, index_obj, data.h_obj_batch, data.h_obj, self.hidden_channels)

        ent_sub = ent_sub.view(data.sub.size(0), self.seq_len, self.hidden_channels)
        ent_obj = ent_obj.view(data.sub.size(0), self.seq_len, self.hidden_channels)

        return ent_sub, ent_obj

    def test(self, logits: Tensor, y: Tensor) -> Tensor:
        """Given ground-truth :obj:`y`, computes Mean Reciprocal Rank (MRR)
        and Hits at 1/3/10."""

        _, perm = logits.sort(dim=1, descending=True)
        mask = (y.view(-1, 1) == perm)

        nnz = mask.nonzero(as_tuple=False)
        mrr = (1 / (nnz[:, -1] + 1).to(torch.float)).mean().item()
        hits1 = mask[:, :1].sum().item() / y.size(0)
        hits2 = mask[:, :2].sum().item() / y.size(0)
        hits3 = mask[:, :3].sum().item() / y.size(0)
        hits10 = mask[:, :10].sum().item() / y.size(0)

        return torch.tensor([mrr, hits1, hits2, hits3, hits10])

    def test_crf(self, logits, y, cluster):
        _, perm = logits.sort(dim=1, descending=True)

        cluster_labels = self.cluster_labels.cuda()[perm]
        cluster = cluster.cuda()

        mask_tags = (cluster_labels == cluster.view(-1, 1)).int()

        first_true_indices_tags = torch.argmax(mask_tags, dim=0)

        mask = (perm == y.view(-1, 1)).int()

        first_true_indices = torch.argmax(mask, dim=0)
        first_true_indices_tags[first_true_indices_tags == 0] = 1000

        hits1 = mask[:, :1].sum().item() / y.size(0) + (first_true_indices_tags == first_true_indices).sum() / len(
            first_true_indices)
        return hits1.item()


class renet_crf(torch.nn.Module):
    def __init__(
            self,
            num_nodes: int,
            num_rels: int,
            hidden_channels: int,
            seq_len: int,
            num_layers: int = 1,
            dropout: float = 0.,
            bias: bool = True,
            num_tags: int = 20
    ):
        super().__init__()
        self.num_tags = num_tags
        self.renet = RENet(num_nodes, num_rels, hidden_channels, seq_len, num_layers, dropout, bias)

    def forward(self, data: Data):
        return self.renet(data)

    def cluster(self):
        self.renet.cluster(self.num_tags)
