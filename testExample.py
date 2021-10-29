from core.layers import EmbeddingLayer
import torch
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
graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
    citation_khop_graph_reconstruction(dataset='citeseer', hop_num=4)
print(graph)
print(node_features.shape)
print(number_of_nodes)
print(number_of_relations)
print(special_entity_dict)
print(special_relation_dict)
print(type(graph))