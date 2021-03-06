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
