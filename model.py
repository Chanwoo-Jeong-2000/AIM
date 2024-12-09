from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index


class DNA_LightGCN(torch.nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int,
                 num_layers: int, alpha: Optional[Union[float, Tensor]] = None, **kwargs):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.embedding = Embedding(num_nodes, embedding_dim)
        self.beta = Embedding(num_nodes, 1) # ours: apply beta
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.beta.weight) # ours: apply beta
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, edge_index: Adj, edge_weight: OptTensor = None,
                      num_users: int = None, num_items: int = None, scaling_factor: float = None) -> Tensor:
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]

        user_emb, item_emb = torch.split(out, [num_users, num_items])
        item_emb = F.normalize(item_emb, p=2, dim=1) * scaling_factor # ours: apply L2-normalization
        out = torch.cat([user_emb, item_emb], dim=0)

        return out

    def forward(self, edge_index: Adj, edge_label_index: OptTensor = None,
                num_users: int = None, num_items: int = None, scaling_factor: float = None,
                edge_weight: OptTensor = None) -> Tensor:
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight,
                                 num_users=num_users, num_items=num_items, scaling_factor=scaling_factor)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        pos_neg_rank = (out_src * out_dst).sum(dim=-1)
        pos_rank, neg_rank = pos_neg_rank.chunk(2)
        pos_item, neg_item = edge_label_index[1].chunk(2)
        pos_rank = pos_rank + self.beta.weight[pos_item] # ours: beta -> involved in BPR
        neg_rank = neg_rank + self.beta.weight[neg_item] # ours: beta -> involved in BPR

        return pos_rank, neg_rank

    def predict_link(self, edge_index: Adj, edge_label_index: OptTensor = None,
                     edge_weight: OptTensor = None, prob: bool = False) -> Tensor:
        pred = self(edge_index, edge_label_index, edge_weight).sigmoid()
        return pred if prob else pred.round()

    def recommend(self, edge_index: Adj, edge_weight: OptTensor = None,src_index: OptTensor = None,
                  dst_index: OptTensor = None, k: int = 1, sorted: bool = True) -> Tensor:
        out_src = out_dst = self.get_embedding(edge_index, edge_weight)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1, sorted=sorted).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor, **kwargs) -> Tensor:
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

    def recommendation_loss(self, pos_edge_rank: Tensor, neg_edge_rank: Tensor,
                            node_id: Optional[Tensor] = None, lambda_reg: float = 1e-4,
                            **kwargs) -> Tensor:
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        emb = self.embedding.weight
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')


class BPRLoss(_Loss):
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor, parameters: Tensor = None) -> Tensor:
        log_prob = F.logsigmoid(positives - negatives).mean()

        regularization = 0
        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)
            regularization = regularization / positives.size(0)

        return -log_prob + regularization