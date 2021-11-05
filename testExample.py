import numpy as np
from dgl.sampling import sample_neighbors
from core.layers import EmbeddingLayer
import torch
from tqdm import tqdm
from time import time
import dgl
from core.kg_utils import KGDataset, knowledge_graph_construction_from_triples, kg_data_path_collection
from core.graph_utils import sub_graph_neighbor_sample, sub_graph_random_walk_sample, cls_sub_graph_extractor
from codes.citation_graph_dataset import citation_graph_reconstruction, citation_khop_graph_reconstruction
from codes.graph_data_loader import SubGraphDataset
from codes.knowledge_graph_dataset import knowledge_graph_khop_reconstruction
from core.utils import seed_everything
from numpy import random
seed_everything(seed=45)
##++++++++++++++
kg_name = 'YAGO3-10'
fanouts = [10,5,5,5]
graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
    knowledge_graph_khop_reconstruction(dataset=kg_name, hop_num=5)
print((graph.in_degrees() == 0).sum())
start_time = time()
kg_dataset = SubGraphDataset(graph=graph, nentity=number_of_nodes, nrelation=number_of_relations,
                                   special_entity2id=special_entity_dict,
                                   special_relation2id=special_relation_dict,
                                   fanouts=fanouts)
for _ in tqdm(range(kg_dataset.len)):
    kg_dataset.__getitem__(_)
print('Run time = {:.4f}'.format(time() - start_time))
##++++++++++++++
# citation_data_name = 'cora'
# graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
#     citation_khop_graph_reconstruction(dataset=citation_data_name, hop_num=6)
# print(special_relation_dict)
# print((graph.in_degrees() == 0).sum())
# start_time = time()
# fanouts = [10,5,5,5,5]
# citation_dataset = SubGraphDataset(graph=graph, nentity=number_of_nodes,
#                                    nrelation=number_of_relations,
#                                    special_entity2id=special_entity_dict,
#                                    special_relation2id=special_relation_dict,
#                                    fanouts=fanouts)
# for _ in tqdm(range(citation_dataset.len)):
#     citation_dataset.__getitem__(_)
#     if _ > 2:
#         break
# print('Run time = {:.4f}'.format(time() - start_time))

# for _ in range(10):
#     y = random.choice(np.arange(10), 1)
#     print(y)
# import scipy.sparse.linalg
# src_ids = torch.tensor([0, 1, 2, 3, 1, 2])
# dst_ids = torch.tensor([0, 1, 2, 3, 2, 1])
# graph = dgl.graph((src_ids, dst_ids), num_nodes=5)
#
# graph = dgl.graph(([], []), num_nodes=5)
# x, y = graph.edges()
#
# z = torch.unique(x, return_counts=True)
# print(z[0].tolist())
#
# cls_nodes = torch.tensor([4] * 4, dtype=torch.long)
# node_ids = torch.arange(4)
#
# graph.add_edges(cls_nodes, node_ids)
# graph.add_edges(node_ids, cls_nodes)
#
# graph_matrix = dgl.khop_adj(graph, 1).numpy()
# diag_maxtrix = np.diag(graph_matrix.sum(axis=1))
# lap_matrix = (diag_maxtrix - graph_matrix)
# norm_lap_matrix = np.eye(5) - np.matmul(np.diag(1.0/graph_matrix.sum(axis=1)), graph_matrix)
# print(norm_lap_matrix)
# # print(graph_matrix)
# # print(lap_matrix)
# x = np.linalg.eig(norm_lap_matrix)
# print(x[0])
# print(x[0].max())
#
# # print(graph)
# # print(dgl.laplacian_lambda_max(graph))
#
# graph.add_edges([3,2], [2,3])
# graph_matrix = dgl.khop_adj(graph, 1).numpy()
# diag_maxtrix = np.diag(graph_matrix.sum(axis=1))
# lap_matrix = (diag_maxtrix - graph_matrix)
# norm_lap_matrix = np.eye(5) - np.matmul(np.diag(1.0/graph_matrix.sum(axis=1)), graph_matrix)
# # print(norm_lap_matrix)
# x = np.linalg.eig(norm_lap_matrix)
# print(x[0])
# print(x[0].max())
# graph_matrix = graph.adjacency_matrix()
# print(graph_matrix)

# print(dgl.laplacian_lambda_max(graph))

# citation_data_name = 'cora'
# graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
#     citation_khop_graph_reconstruction(dataset=citation_data_name, hop_num=6)

# x = dgl.khop_graph(graph, 5)
# y = dgl.out_subgraph(x, torch.arange(x.number_of_nodes()))
# # print(x)
# print(dgl.khop_adj(x, 1))
# print(dgl.khop_adj(y, 1))

# number_of_nodes = graph.number_of_nodes()
# print(number_of_nodes)
# node_ids = torch.arange(number_of_nodes - 1)
# self_loop_r = torch.LongTensor(number_of_nodes - 1).fill_(6)
# graph.add_edges(node_ids, node_ids, {'tid': self_loop_r})
# print(graph)
# anchor_nodes = torch.tensor([0], dtype=torch.long)
# cls_nodes = torch.tensor([5], dtype=torch.long)
# sg = sample_neighbors(g=graph, nodes=[], edge_dir='in', fanout=4)
# print(sg)
# neighbors_dict, edge_dict = sub_graph_neighbor_sample(graph=graph, fanouts=[4,4,4], edge_dir='out',
#                           anchor_node_ids=anchor_nodes, cls_node_ids=cls_nodes)
# print(neighbors_dict)
# print(edge_dict)
# print(sg)
# sg_src, sg_dst = sg.edges()
# print(sg_src.shape[0])
# print(sg_dst)