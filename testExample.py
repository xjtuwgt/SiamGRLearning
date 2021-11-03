import numpy as np
from dgl.sampling import sample_neighbors
from core.layers import EmbeddingLayer
import torch
from tqdm import tqdm
from time import time
import dgl
from core.kg_utils import KGDataset, knowledge_graph_construction_from_triples, kg_data_path_collection
from core.graph_utils import sub_graph_neighbor_sample, sub_graph_neighbor_sample_unique, \
    sub_graph_random_walk_sample, cls_sub_graph_extractor
from codes.citation_graph_dataset import citation_graph_reconstruction, citation_khop_graph_reconstruction
from codes.graph_data_loader import SubGraphDataset
from codes.knowledge_graph_dataset import knowledge_graph_khop_reconstruction
from core.utils import seed_everything
from numpy import random
seed_everything(seed=45)

##++++++++++++++
# kg_name = 'FB15k-237'
# fanouts = [10,5,5,5]
# graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
#     knowledge_graph_khop_reconstruction(dataset=kg_name, hop_num=5)
# print((graph.in_degrees() == 0).sum())
# start_time = time()
# kg_dataset = SubGraphDataset(graph=graph, nentity=number_of_nodes, nrelation=number_of_relations,
#                                    special_entity2id=special_entity_dict,
#                                    special_relation2id=special_relation_dict,
#                                    fanouts=fanouts)
# for _ in tqdm(range(kg_dataset.len)):
#     kg_dataset.__getitem__(_)
# print('Run time = {:.4f}'.format(time() - start_time))
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
src_ids = torch.tensor([2, 3, 4])
dst_ids = torch.tensor([0, 1, 2])
graph = dgl.graph((src_ids, dst_ids))
graph.edata['tid'] = torch.arange(0, 3)

number_of_nodes = graph.number_of_nodes()
print(number_of_nodes)
node_ids = torch.arange(number_of_nodes - 1)
self_loop_r = torch.LongTensor(number_of_nodes - 1).fill_(6)
graph.add_edges(node_ids, node_ids, {'tid': self_loop_r})
print(graph)
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