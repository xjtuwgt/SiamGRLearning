from dgl import DGLHeteroGraph
from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F

"""
    If the number of neighbors is large, then we need to select a subset of neighbors for feature aggregation
    codes for mask testing
    def edge_mask_message_func(edges):
        return {'m_a_mask': edges.data['attn_mask']}

    def mask_top_k_reduce_func(nodes):
        edge_attention_mask = nodes.mailbox['m_a_mask']
        res = edge_attention_mask.sum(1)
        print(torch.min(res.squeeze(-1)))
        # print(res.shape)
        return {'k_num': res}

    graph.update_all(edge_mask_message_func, mask_top_k_reduce_func)
"""


def edge_message_func(edges):
    return {'m_a': edges.data['ta']}


def edge_udf_attn_func(edges):
    attention_scores = edges.data['ta']
    top_k_attn_score = edges.dst['top_a']
    attention_mask = attention_scores >= top_k_attn_score
    return {'attn_mask': attention_mask}


def top_k_attention(graph: DGLHeteroGraph, attn_scores: Tensor, k: int = 5):
    """
    :param attn_scores:
    :param graph:
    :param k:
    :return:
    """

    def top_k_reduce_func(nodes):
        edge_attention_score = nodes.mailbox['m_a']
        batch_size, neighbor_num, head_num, _ = edge_attention_score.shape
        if neighbor_num <= k:
            ret_a = torch.empty(batch_size, head_num, 1).fill_(edge_attention_score.min())
            ret_a_sum = edge_attention_score.sum(dim=1)
        else:
            top_k_values, _ = torch.topk(edge_attention_score, k=k, dim=1)
            ret_a = top_k_values[:, -1, :, :]
            ret_a_sum = top_k_values.sum(dim=1)
        return {'top_a': ret_a, 'top_as': ret_a_sum}

    with graph.local_scope():
        graph.edata['ta'] = attn_scores
        graph.update_all(edge_message_func, top_k_reduce_func)
        graph.apply_edges(edge_udf_attn_func)
        attn_mask = graph.edata.pop('attn_mask')
        top_k_attn_sum = graph.ndata.pop('top_as')
        return attn_mask, top_k_attn_sum


def top_p_attention(graph: DGLHeteroGraph, attn_scores: Tensor, p: float = 0.75):
    """
    when the distribution attention values are quite imbalanced (especially for peak distribution)
    :param attn_scores:
    :param graph:
    :param p:
    :return:
    """

    def top_p_reduce_func(nodes):
        edge_attention_score = nodes.mailbox['m_a']
        batch_size, neighbor_num, head_num, _ = edge_attention_score.shape
        sorted_attends, sorted_indices = torch.sort(edge_attention_score, descending=True, dim=1)
        cumulative_attends = torch.cumsum(sorted_attends, dim=1)
        neighbors_to_remove = cumulative_attends > p
        remove_neighbor_count = neighbors_to_remove.sum(dim=1)
        keep_neighbor_count = neighbor_num - remove_neighbor_count
        assert keep_neighbor_count.min() >= 0 and keep_neighbor_count.max() <= neighbor_num
        if keep_neighbor_count.min() == 0:
            keep_neighbor_count[keep_neighbor_count == 0] = 1
        top_p_score_idx = (keep_neighbor_count - 1)
        assert top_p_score_idx.max() < neighbor_num
        sorted_attends = sorted_attends.squeeze(-1).transpose(1, 2).contiguous().view(batch_size * head_num,
                                                                                      neighbor_num)
        top_p_score_idx = top_p_score_idx.squeeze(-1).contiguous().view(batch_size * head_num)
        row_arrange_idx = torch.arange(batch_size * head_num)
        ret_a = sorted_attends[row_arrange_idx, top_p_score_idx].view(batch_size, head_num, 1)
        cumulative_attends = cumulative_attends.squeeze(-1).transpose(1, 2).contiguous().view(batch_size * head_num,
                                                                                              neighbor_num)
        ret_a_sum = cumulative_attends[row_arrange_idx, top_p_score_idx].view(batch_size, head_num, 1)
        return {'top_a': ret_a, 'top_as': ret_a_sum}

    with graph.local_scope():
        graph.edata['ta'] = attn_scores
        graph.update_all(edge_message_func, top_p_reduce_func)
        graph.apply_edges(edge_udf_attn_func)
        attn_mask = graph.edata.pop('attn_mask')
        top_k_attn_sum = graph.ndata.pop('top_as')
        return attn_mask, top_k_attn_sum


def top_kp_attention(graph: DGLHeteroGraph, attn_scores: Tensor, k: int = 5, p: float = 0.75, sparse_mode='top_k'):
    if sparse_mode == 'top_k':
        assert k >= 2
        attn_mask, top_k_attn_sum = top_k_attention(graph=graph, attn_scores=attn_scores, k=k)
    else:
        assert 0.0 < p <= 1.0
        attn_mask, top_k_attn_sum = top_p_attention(graph=graph, attn_scores=attn_scores, p=p)
    return attn_mask, top_k_attn_sum


def top_kp_attn_normalization(graph: DGLHeteroGraph, attn_scores: Tensor, attn_mask: Tensor, top_k_sum: Tensor):
    def edge_udf_attn_normalization_func(edges):
        attention_scores = edges.data['ta']
        sum_attn_scores = edges.dst['attn_sum']
        norm_attends = attention_scores / sum_attn_scores
        return {'norm_attn': norm_attends}

    with graph.local_scope():
        graph.edata['ta'] = attn_scores
        graph.edata['ta'][~attn_mask] = 0.0
        graph.dstdata['attn_sum'] = top_k_sum
        graph.apply_edges(edge_udf_attn_normalization_func)
        norm_attentions = graph.edata.pop('norm_attn')
        return norm_attentions


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, model_dim, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.model_dim = model_dim
        self.hidden_dim = d_hidden
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def init(self):
        # gain = nn.init.calculate_gain('relu')
        gain = small_init_gain(d_in=self.model_dim, d_out=self.hidden_dim)
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)
        gain = small_init_gain(d_in=self.hidden_dim, d_out=self.model_dim)
        nn.init.xavier_normal_(self.w_2.weight, gain=gain)


def small_init_gain(d_in, d_out):
    return 2.0 / (d_in + 4.0 * d_out)