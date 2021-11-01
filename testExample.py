from core.layers import EmbeddingLayer
import torch
import dgl
from core.kg_utils import KGDataset, knowledge_graph_construction_from_triples, kg_data_path_collection
from core.graph_utils import graph_to_bidirected
from core.graph_utils import sub_graph_neighbor_sample, sub_graph_neighbor_sample_unique, \
    sub_graph_random_walk_sample, sub_graph_extractor, cls_sub_graph_extractor
from codes.citation_graph_dataset import citation_graph_reconstruction, citation_khop_graph_reconstruction
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
from core.utils import seed_everything
from codes.kg_dataset import knowledge_graph_khop_reconstruction
seed_everything(seed=42)
kg_name = 'wn18rr'
graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
    knowledge_graph_khop_reconstruction(dataset=kg_name, hop_num=5)

print(number_of_relations, number_of_nodes)
anchor = torch.LongTensor([0])
cls = torch.LongTensor([special_entity_dict['cls']])
x, y = sub_graph_neighbor_sample(graph=graph, anchor_node_ids=anchor, cls_node_ids=cls,
                           fanouts=[5,5,5,5,5], edge_dir='in', debug=True)
print(len(y))
# sub_graph = sub_graph_extractor(graph=graph, edge_dict=y)
cls_sub_graph_extractor(graph=graph, edge_dict=y, neighbors_dict=x, special_relation_dict=special_relation_dict,
                        bi_directed=True)

# x, y = sub_graph_random_walk_sample(graph=graph, anchor_node_ids=anchor, cls_node_ids=cls,
#                            fanouts=[4] * 5, edge_dir='in', debug=True)
# print(len(y))

# if __name__ == '__main__':
#     import os
#     from evens import KG_DATA_FOLDER
#     kg_name = 'wn18rr'
#     entity_path, relation_path, train_path, _, _ = kg_data_path_collection(kg_path=KG_DATA_FOLDER, kg_name=kg_name)
#     kg_data = KGDataset(entity_path=entity_path, relation_path=relation_path,
#                         train_path=train_path)
#     # train_data = kg_data.train
#     # print(train_data)
#     # print(kg_data.entity2id)
#     # for k, v in kg_data.relation2id.items():
#     #     print(k, v)
#     graph = knowledge_graph_construction_from_triples(num_entities=kg_data.n_entities,
#                                                       num_relations=kg_data.n_relations,
#                                                       triples=kg_data.train, bi_directional=False)
#     print(graph)
#
#     # anchor = torch.LongTensor([0])
#     # cls = torch.LongTensor([special_entity_dict['cls']])
#     # x, y = sub_graph_neighbor_sample(graph=graph, anchor_node_ids=anchor, cls_node_ids=cls,
#     #                            fanouts=[5,5,5,5,5], edge_dir='out', debug=True)