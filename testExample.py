# import numpy as np
# from dgl.sampling import sample_neighbors
# from core.layers import EmbeddingLayer
# from ogb.nodeproppred import DglNodePropPredDataset
# from evens import HOME_DATA_FOLDER
# import torch
# from tqdm import tqdm
# from time import time
# import dgl
# from core.kg_utils import KGDataset, knowledge_graph_construction_from_triples, kg_data_path_collection
# from core.graph_utils import sub_graph_neighbor_sample, sub_graph_random_walk_sample, cls_sub_graph_extractor
# from codes.citation_graph_data import citation_graph_reconstruction, citation_khop_graph_reconstruction
# from codes.graph_pretrained_dataset import NodeSubGraphPairDataset
# from codes.ogb_graph_data import ogb_khop_graph_reconstruction
# from codes.graph_train_dataset import NodePredSubGraphDataset, NodeSubGraphDataset
# from codes.knowledge_graph_data import knowledge_graph_khop_reconstruction
# from core.utils import seed_everything
# from torch.utils.data import DataLoader
# from numpy import random
# seed_everything(seed=45)
# ##++++++++++++++
# # kg_name = 'FB15k-237'
# # fanouts = [15,10,5,5]
# # graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
# #     knowledge_graph_khop_reconstruction(dataset=kg_name, hop_num=5)
# # print((graph.in_degrees() == 0).sum())
# # start_time = time()
# # kg_dataset = SubGraphPairDataset(graph=graph, nentity=number_of_nodes, nrelation=number_of_relations,
# #                                    special_entity2id=special_entity_dict,
# #                                    special_relation2id=special_relation_dict,
# #                                    fanouts=fanouts)
# # kg_dataloader = DataLoader(dataset=kg_dataset,
# #                                  batch_size=16,
# #                                  collate_fn=SubGraphPairDataset.collate_fn)
# # # for _ in tqdm(range(kg_dataset.len)):
# # #     kg_dataset.__getitem__(_)
# # for batch_idx, batch in tqdm(enumerate(kg_dataloader)):
# #     batch_graph, batch_cls = batch['batch_graph_1']
# #     print(batch_graph.ndata['nid'][batch_cls], batch_graph.number_of_nodes())
# # print('Run time = {:.4f}'.format(time() - start_time))
# # ##++++++++++++++
# citation_data_name = 'citeseer'
# graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, \
# special_relation_dict, n_classes, n_feats = \
#     citation_khop_graph_reconstruction(dataset=citation_data_name, hop_num=6)
# print(graph.ndata)
# print('Number of nodes with 0 in-degree = {}'.format((graph.in_degrees() == 0).sum()))
# print(graph.number_of_nodes())
# print(graph.ndata['label'])
# print(graph.ndata)
# start_time = time()
# fanouts = [10,5,5,5]
# # fanouts = [-1,-1,-1,-1]
# # citation_dataset = SubGraphPairDataset(graph=graph, nentity=number_of_nodes,
# #                                    nrelation=number_of_relations,
# #                                    special_entity2id=special_entity_dict,
# #                                    special_relation2id=special_relation_dict,
# #                                    fanouts=fanouts)
# # citation_dataloader = DataLoader(dataset=citation_dataset,
# #                                  batch_size=16,
# #                                  collate_fn=SubGraphPairDataset.collate_fn)
# # for _ in tqdm(range(citation_dataset.len)):
# #     citation_dataset.__getitem__(_)
# #     break
#
# citation_dataset = NodePredSubGraphDataset(graph=graph, nentity=number_of_nodes,
#                                    nrelation=number_of_relations,
#                                    special_entity2id=special_entity_dict,
#                                    special_relation2id=special_relation_dict,
#                                    fanouts=fanouts, graph_type='citation', data_type='train')
# citation_dataloader = DataLoader(dataset=citation_dataset,
#                                  batch_size=16,
#                                  collate_fn=NodeSubGraphPairDataset.collate_fn)
# for _ in tqdm(range(citation_dataset.len)):
#     citation_dataset.__getitem__(_)
#
#
# # # print('Run time = {:.4f}'.format(time() - start_time))
# # for batch_idx, batch in tqdm(enumerate(citation_dataloader)):
# #     batch_graph, batch_cls = batch['batch_graph_2']
# #     # print(batch_graph.ndata['nid'][batch_cls])
# # print('Run time = {:.4f}'.format(time() - start_time))
#
# # ogb_dataname = 'ogbn-arxiv'
# #
# # data = DglNodePropPredDataset(name=ogb_dataname, root=HOME_DATA_FOLDER)
# # graph, labels = data[0]
# # print(graph.number_of_nodes())
# # print(labels.shape)
# # graph, node_split_idx, node_features, number_of_nodes, number_of_relations, special_entity_dict,\
# # special_relation_dict, n_classes, n_feats = \
# #         ogb_khop_graph_reconstruction(dataset=ogb_dataname, hop_num=6)
# # print('Number of nodes = {}'.format(number_of_nodes))
# # print('Number of eges = {}'.format(graph.number_of_edges()))
# # print('Node features = {}'.format(n_feats))
# # print('Number of relations = {}'.format(number_of_relations))
# # print('Number of nodes with 0 in-degree = {}'.format((graph.in_degrees() == 0).sum()))
# # fanouts = [10, 5, 5]
# #
# # print(graph.number_of_edges())
# #
# #
# # ogb_dataset = SubGraphPairDataset(graph=graph, nentity=number_of_nodes, nrelation=number_of_relations,
# #                                   special_entity2id=special_entity_dict,
# #                                   special_relation2id=special_relation_dict, fanouts=fanouts)
# # #
# # for _ in tqdm(range(ogb_dataset.len)):
# #     ogb_dataset.__getitem__(_)
#
#
# # def logbook_solution(logbook: list):
# #     char_dict = {}
# #     for ch_1, ch_2 in logbook:
# #         start_i = ord(ch_1.upper())
# #         end_i = ord(ch_2.upper())
# #         if start_i > end_i:
# #             temp = start_i
# #             start_i = end_i
# #             end_i = temp
# #         if start_i not in char_dict:
# #             char_dict[start_i] = end_i
# #         else:
# #             if end_i > char_dict[start_i]:
# #                 char_dict[start_i] = end_i
# #     # ++++++++++++++++++++++++++++++++++++++++++++++++++
# #     # dictionary will reduce the number of possible pairs in logbook
# #     # ++++++++++++++++++++++++++++++++++++++++++++++++++
# #     sorted_keys = sorted(char_dict)
# #     prev_start = sorted_keys[0]
# #     prev_right = char_dict[prev_start]
# #     max_len = prev_right - prev_start
# #     for i in range(1, len(sorted_keys)):
# #         if sorted_keys[i] <= prev_right:
# #             prev_right = char_dict[sorted_keys[i]]
# #         else:
# #             prev_start = sorted_keys[i]
# #             prev_right = char_dict[sorted_keys[i]]
# #         cur_len = prev_right - prev_start
# #         if max_len <= cur_len:
# #             max_len = cur_len
# #     return max_len
# #
# #
# # def solution(logbook):
# #     global_max = 0
# #     sorted_log = []
# #     for log in logbook:
# #         if log[0] > log[1]:
# #             sorted_log.append(log[1] + log[0])
# #         else:
# #             sorted_log.append(log)
# #
# #     logbook = sorted(sorted_log)
# #     prev_right = logbook[0][1]
# #     prev_start = logbook[0][0]
# #     for log in logbook:
# #         if log[0] <= prev_right:
# #             cur_len = max(ord(log[1]), ord(prev_right)) - ord(prev_start)
# #             global_max = max(global_max, cur_len)
# #             prev_right = max(prev_right, log[1])
# #         else:
# #             prev_start = log[0]
# #             prev_right = log[1]
# #             # ++++
# #             cur_len = ord(prev_right) - ord(prev_start)
# #             global_max = max(global_max, cur_len)
# #             # ++++
# #     return global_max
# #
# # # def sqrt(x):
# # #     last_guess = x/2.0
# # #     while True:
# # #         guess = (last_guess + x/last_guess)/2
# # #         if abs(guess - last_guess) < .000001: # example threshold
# # #             return guess
# # #         last_guess = guess
# # #
# # #
# # # def sqrt_solution(predicted, observed):
# # #     # write your code in Python 3.6
# # #     n = len(predicted)
# # #     sum_se = 0
# # #     for i in range(n):
# # #         sum_se += (predicted[i] - observed[i]) ** 2
# # #
# # #     avg_se = sum_se / n
# # #     print(avg_se)
# # #
# # #     def square(n, left, right):
# # #         mid = (left + right) / 2
# # #         square_sum = mid * mid
# # #         if (square_sum == n) or (abs(square_sum - n) < 0.000001):
# # #             return mid
# # #         elif (square_sum < n):
# # #             return square(n, mid, right)
# # #         else:
# # #             return square(n, left, mid)
# # #
# # #     def find_square_root(n):
# # #         left = 1
# # #         Ind = False
# # #         while (Ind == False):
# # #             if (left * left == n):
# # #                 res = left
# # #                 Ind = True
# # #             elif (left * left > n):
# # #                 res = square(n, left - 1, left)
# # #                 Ind = True
# # #             left += 1
# # #         return res
# # #     rmse = find_square_root(avg_se)
# # #     return rmse
# # #
# # # def sqrt_solution(predicted, observed):
# # #     # write your code in Python 3.6
# # #     n = len(predicted)
# # #     sum_se = 0
# # #     for i in range(n):
# # #         sum_se += (predicted[i] - observed[i]) ** 2
# # #
# # #     avg_se = sum_se / n
# # #
# # #     def find_square_root(x):
# # #         last_guess = x / 2.0
# # #         while True:
# # #             guess = (last_guess + x / last_guess) / 2
# # #             if abs(guess - last_guess) < .000001:  # example threshold
# # #                 return guess
# # #             last_guess = guess
# # #     rmse = find_square_root(avg_se)
# # #     return rmse
# #
# # if __name__ == '__main__':
# #     print()
# #     # x = ['BG', 'CA', 'FI', 'OK']
# #     # x = ['BG', 'CA']
# #     # x = ['FI', 'OK']
# #     x = ['AB', 'AC', 'AD', 'OK']
# #     # # x = ['AB', 'CF']
# #     # # x = ['AB', 'AC', 'CF']
# #     # # x = ['ab', 'aF']
# #     y = logbook_solution(x)
# #     print(y)
# #     y = solution(x)
# #     print(y)
# #     # y = sqrt(3)
# #     # print(y)
# #
# #     # x = [4, 25, 0.75, 11]
# #     # y = [3, 21, -1.25, 13]
# #     #
# #     # z = sqrt_solution(x, y)
# #     # print(z)
# #
# #


if __name__ == '__main__':
    import torch
    import dgl
    import torch.nn as nn
    from dgl.data import CoraGraphDataset
    import dgl.function as fn
    import torch as th
    from torch import nn
    import torch
    from core.utils import seed_everything
    from core.gnn_utils import top_kp_attention, top_kp_attn_normalization

    import torch.nn.functional as F
    from dgl.nn.pytorch.utils import Identity
    import dgl.function as fn
    from torch import Tensor
    from dgl.nn.functional import edge_softmax
    from torch.nn import LayerNorm as layer_norm
    from dgl.base import DGLError
    from dgl.utils import expand_as_pair

    seed_everything(seed=42)


    # pylint: enable=W0235
    class GATConv(nn.Module):
        r"""
        Description
        -----------
        Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
        over an input signal.
        .. math::
            h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
        where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
        node :math:`j`:
        .. math::
            \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})
            e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)
        Parameters
        ----------
        in_feats : int, or pair of ints
            Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
            GATConv can be applied on homogeneous graph and unidirectional
            `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
            If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
            specifies the input feature size on both the source and destination nodes.  If
            a scalar is given, the source and destination node feature size would take the
            same value.
        out_feats : int
            Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
        num_heads : int
            Number of heads in Multi-Head Attention.
        feat_drop : float, optional
            Dropout rate on feature. Defaults: ``0``.
        attn_drop : float, optional
            Dropout rate on attention weight. Defaults: ``0``.
        negative_slope : float, optional
            LeakyReLU angle of negative slope. Defaults: ``0.2``.
        residual : bool, optional
            If True, use residual connection. Defaults: ``False``.
        activation : callable activation function/layer or None, optional.
            If not None, applies an activation function to the updated node features.
            Default: ``None``.
        allow_zero_in_degree : bool, optional
            If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
            since no message will be passed to those nodes. This is harmful for some applications
            causing silent performance regression. This module will raise a DGLError if it detects
            0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
            and let the users handle it by themselves. Defaults: ``False``.
        bias : bool, optional
            If True, learns a bias term. Defaults: ``True``.
        Note
        ----
        Zero in-degree nodes will lead to invalid output value. This is because no message
        will be passed to those nodes, the aggregation function will be appied on empty input.
        A common practice to avoid this is to add a self-loop for each node in the graph if
        it is homogeneous, which can be achieved by:
        tensor([[[-0.6066,  1.0268],
                [-0.5945, -0.4801],
                [ 0.1594,  0.3825]],
                [[ 0.0268,  1.0783],
                [ 0.5041, -1.3025],
                [ 0.6568,  0.7048]],
                [[-0.2688,  1.0543],
                [-0.0315, -0.9016],
                [ 0.3943,  0.5347]],
                [[-0.6066,  1.0268],
                [-0.5945, -0.4801],
                [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
        """

        def __init__(self,
                     in_feats,
                     out_feats,
                     num_heads,
                     feat_drop=0.,
                     attn_drop=0.,
                     negative_slope=0.2,
                     residual=False,
                     activation=None,
                     allow_zero_in_degree=False,
                     bias=True):
            super(GATConv, self).__init__()
            self._num_heads = num_heads
            self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
            self._out_feats = out_feats
            self._allow_zero_in_degree = allow_zero_in_degree
            if isinstance(in_feats, tuple):
                self.fc_src = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False)
                self.fc_dst = nn.Linear(
                    self._in_dst_feats, out_feats * num_heads, bias=False)
            else:
                self.fc = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False)
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            self.feat_drop = nn.Dropout(feat_drop)
            self.attn_drop = nn.Dropout(attn_drop)
            self.leaky_relu = nn.LeakyReLU(negative_slope)
            if bias:
                self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
            else:
                self.register_buffer('bias', None)
            if residual:
                if self._in_dst_feats != out_feats * num_heads:
                    self.res_fc = nn.Linear(
                        self._in_dst_feats, num_heads * out_feats, bias=False)
                else:
                    self.res_fc = Identity()
            else:
                self.register_buffer('res_fc', None)
            self.reset_parameters()
            self.activation = activation

        def reset_parameters(self):
            """
            Description
            -----------
            Reinitialize learnable parameters.
            Note
            ----
            The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
            The attention weights are using xavier initialization method.
            """
            gain = nn.init.calculate_gain('relu')
            if hasattr(self, 'fc'):
                nn.init.xavier_normal_(self.fc.weight, gain=gain)
            else:
                nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
                nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.attn_l, gain=gain)
            nn.init.xavier_normal_(self.attn_r, gain=gain)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
            if isinstance(self.res_fc, nn.Linear):
                nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

        def set_allow_zero_in_degree(self, set_value):
            r"""
            Description
            -----------
            Set allow_zero_in_degree flag.
            Parameters
            ----------
            set_value : bool
                The value to be set to the flag.
            """
            self._allow_zero_in_degree = set_value

        def forward(self, graph, feat, get_attention=False):
            r"""
            Description
            -----------
            Compute graph attention network layer.
            Parameters
            ----------
            graph : DGLGraph
                The graph.
            feat : torch.Tensor or pair of torch.Tensor
                If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
                :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
                If a pair of torch.Tensor is given, the pair must contain two tensors of shape
                :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
            get_attention : bool, optional
                Whether to return the attention values. Default to False.
            Returns
            -------
            torch.Tensor
                The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
                is the number of heads, and :math:`D_{out}` is size of output feature.
            torch.Tensor, optional
                The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
                edges. This is returned only when :attr:`get_attention` is ``True``.
            Raises
            ------
            DGLError
                If there are 0-in-degree nodes in the input graph, it will raise DGLError
                since no message will be passed to those nodes. This will cause invalid output.
                The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
            """
            with graph.local_scope():
                if not self._allow_zero_in_degree:
                    if (graph.in_degrees() == 0).any():
                        raise DGLError('There are 0-in-degree nodes in the graph, '
                                       'output for those nodes will be invalid. '
                                       'This is harmful for some applications, '
                                       'causing silent performance regression. '
                                       'Adding self-loop on the input graph by '
                                       'calling `g = dgl.add_self_loop(g)` will resolve '
                                       'the issue. Setting ``allow_zero_in_degree`` '
                                       'to be `True` when constructing this module will '
                                       'suppress the check and let the code run.')

                if isinstance(feat, tuple):
                    src_prefix_shape = feat[0].shape[:-1]
                    dst_prefix_shape = feat[1].shape[:-1]
                    h_src = self.feat_drop(feat[0])
                    h_dst = self.feat_drop(feat[1])
                    if not hasattr(self, 'fc_src'):
                        feat_src = self.fc(h_src).view(
                            *src_prefix_shape, self._num_heads, self._out_feats)
                        feat_dst = self.fc(h_dst).view(
                            *dst_prefix_shape, self._num_heads, self._out_feats)
                    else:
                        feat_src = self.fc_src(h_src).view(
                            *src_prefix_shape, self._num_heads, self._out_feats)
                        feat_dst = self.fc_dst(h_dst).view(
                            *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                    h_src = h_dst = self.feat_drop(feat)
                    feat_src = feat_dst = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    if graph.is_block:
                        feat_dst = feat_src[:graph.number_of_dst_nodes()]
                        h_dst = h_dst[:graph.number_of_dst_nodes()]
                        dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
                # NOTE: GAT paper uses "first concatenation then linear projection"
                # to compute attention scores, while ours is "first projection then
                # addition", the two approaches are mathematically equivalent:
                # We decompose the weight vector a mentioned in the paper into
                # [a_l || a_r], then
                # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
                # Our implementation is much efficient because we do not need to
                # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
                # addition could be optimized with DGL's built-in function u_add_v,
                # which further speeds up computation and saves memory footprint.
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.srcdata.update({'ft': feat_src, 'el': el})
                graph.dstdata.update({'er': er})
                # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                e = self.leaky_relu(graph.edata.pop('e'))
                # compute softmax
                a = edge_softmax(graph, e)
                print((a==0).sum(), graph.number_of_edges() * self._num_heads)
                graph.edata['ta'] = a
                a_mask, a_top_sum = top_kp_attention(graph=graph, attn_scores=a, top_k=True)
                n_a = top_kp_attn_normalization(graph=graph, attn_scores=a, attn_mask=a_mask, topk_sum=a_top_sum)
                print((n_a == 0).sum(), graph.number_of_edges() * self._num_heads)

                graph.edata['a'] = self.attn_drop(a)
                # message passing
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                 fn.sum('m', 'ft'))
                rst = graph.dstdata['ft']
                # residual
                if self.res_fc is not None:
                    # Use -1 rather than self._num_heads to handle broadcasting
                    resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                    rst = rst + resval
                # bias
                if self.bias is not None:
                    rst = rst + self.bias.view(
                        *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
                # activation
                if self.activation:
                    rst = self.activation(rst)

                if get_attention:
                    return rst, graph.edata['a']
                else:
                    return rst






    class GAT(nn.Module):
        def __init__(self,
                     g,
                     num_layers,
                     in_dim,
                     num_hidden,
                     num_classes,
                     heads,
                     activation,
                     feat_drop,
                     attn_drop,
                     negative_slope,
                     residual):
            super(GAT, self).__init__()
            self.g = g
            self.num_layers = num_layers
            self.gat_layers = nn.ModuleList()
            self.activation = activation
            # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * heads[l - 1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))

        def forward(self, inputs):
            h = inputs
            for l in range(self.num_layers):
                h = self.gat_layers[l](self.g, h).flatten(1)
            # output projection
            logits = self.gat_layers[-1](self.g, h).mean(1)
            return logits


    data = CoraGraphDataset()
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    heads = ([8] * 1) + [8]
    model = GAT(g,
                1,
                num_feats,
                32,
                n_classes,
                heads,
                F.elu,
                0.1,
                0.2,
                0.2,
                True)
    print(model)

    logits = model(features)
    # print(logits)
    print(logits.shape)
