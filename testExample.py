from core.layers import EmbeddingLayer
import torch
import dgl
from core.graph_utils import graph_to_bidirected
from core.graph_utils import sub_graph_neighbor_sample, sub_graph_neighbor_sample_unique, sub_graph_random_walk_sample
from codes.graph_dataset import citation_graph_reconstruction, citation_khop_graph_reconstruction
# from dgl.data import CoraGraphDataset
#
# embed_layer = EmbeddingLayer(num=10, dim=3)
# idxes = torch.LongTensor([1,2,3])
#
# print(embed_layer)
# print(embed_layer(idxes).shape)
#
# graph = CoraGraphDataset()
# # print(graph)
# x = list(graph[0].ndata.keys())
# print(x)
# print(type(x))

# output = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long))
# print(output)

# graph, node_features, nentities, nrelations = citation_graph_reconstruction(dataset='cora')
# print(graph)
# print(node_features.shape)
# print(nentities, nrelations)
# graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
#     citation_khop_graph_reconstruction(dataset='cora', hop_num=4)
# print(graph)
# print(node_features.shape)
# print(number_of_nodes)
# print(number_of_relations)
# print(special_entity_dict)
# print(special_relation_dict)
# print(type(graph))

# print(type(dgl.EID), dgl.ETYPE)
# g = dgl.graph((torch.tensor([0, 0, 2, 1, 1, 1]), torch.tensor([1, 0, 0, 2, 3, 2])))
# g.edata['tid'] = torch.tensor([0]*g.number_of_edges(), dtype=torch.long)
# g1 = dgl.remove_self_loop(g)
# print(g1)
#
# g1 = graph_to_bidirected(graph=g1, is_hete_graph=True, number_of_relations=1)
# print(g1)
# print(g1.edges('all'))
# r_g = dgl.reverse(g=g1, copy_ndata=True, copy_edata=True)
# print(r_g)
# print(r_g.edges('all'))
# src_nodes, dst_nodes, edge_ids = g.edges(form='all')
# print(src_nodes)
# print(dst_nodes)
# print(edge_ids)

# g1 = dgl.to_bidirected(g)
# print(g1)
# # print(g.edata[dgl.EID])
# g.add_nodes(2)
# g.edata['tid'] = torch.zeros(g.number_of_edges(), dtype=torch.long)
#
# anchor = torch.LongTensor([0])
# cls = torch.LongTensor([special_entity_dict['cls']])
# x, y = sub_graph_neighbor_sample(graph=graph, anchor_node_ids=anchor, cls_node_ids=cls,
#                            fanouts=[5,5,5,5,5], edge_dir='out', debug=True)
# # print(x)
# # print(y)
# # print(z)
# #
# x, y = sub_graph_neighbor_sample_unique(graph=graph, anchor_node_ids=anchor, cls_node_ids=cls,
#                            fanouts=[5,5,5,5,5], edge_dir='out', debug=True)
# # # print(x)
# # # print(y)
# # # print(z)
# #
# sub_graph_random_walk_sample(graph=graph, anchor_node_ids=anchor, cls_node_ids=cls,
#                            fanouts=[5,5,5,5,5], edge_dir='out', debug=True)

# x = g.has_edges_between([1,2], [2,3])
# print(x)
# y = g.edge_ids(1, 2)
# print(y)