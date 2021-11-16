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
# kg_name = 'wn18rr'
# fanouts = [-1,5,5]
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
citation_data_name = 'cora'
graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
    citation_khop_graph_reconstruction(dataset=citation_data_name, hop_num=6)
print('Number of nodes with 0 in-degree = {}'.format((graph.in_degrees() == 0).sum()))
start_time = time()
fanouts = [10,5,5,5]
# fanouts = [-1,-1,-1,-1]
citation_dataset = SubGraphDataset(graph=graph, nentity=number_of_nodes,
                                   nrelation=number_of_relations,
                                   special_entity2id=special_entity_dict,
                                   special_relation2id=special_relation_dict,
                                   fanouts=fanouts)
for _ in tqdm(range(citation_dataset.len)):
    citation_dataset.__getitem__(_)
print('Run time = {:.4f}'.format(time() - start_time))
