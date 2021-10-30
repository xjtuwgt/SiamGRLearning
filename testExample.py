from core.layers import EmbeddingLayer
import torch
import dgl
from core.graph_utils import sub_graph_sample
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

# graph, node_features, nentities, nrelations = citation_graph_reconstruction(dataset='cora')
# print(graph)
# print(node_features.shape)
# print(nentities, nrelations)
# graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
#     citation_khop_graph_reconstruction(dataset='citeseer', hop_num=4)
# print(graph)
# print(node_features.shape)
# print(number_of_nodes)
# print(number_of_relations)
# print(special_entity_dict)
# print(special_relation_dict)
# print(type(graph))


g = dgl.graph((torch.tensor([0, 0, 2, 1, 1, 1]), torch.tensor([1, 0, 0, 2, 3, 2])))
g.add_nodes(2)
g.edata['tid'] = torch.zeros(g.number_of_edges(), dtype=torch.long)

anchor = torch.LongTensor([0])
cls = torch.LongTensor([4])
x, y, z = sub_graph_sample(graph=g, anchor_node_ids=anchor, cls_node_ids=cls, fanouts=[2,2], edge_dir='out', bi_direct=True)
print(x)
print(y)
print(z)

# x = g.has_edges_between([1,2], [2,3])
# print(x)
# y = g.edge_ids(1, 2)
# print(y)