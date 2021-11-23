import numpy as np
from dgl.sampling import sample_neighbors
from core.layers import EmbeddingLayer
from ogb.nodeproppred import DglNodePropPredDataset
from evens import HOME_DATA_FOLDER
import torch
from tqdm import tqdm
from time import time
import dgl
from core.kg_utils import KGDataset, knowledge_graph_construction_from_triples, kg_data_path_collection
from core.graph_utils import sub_graph_neighbor_sample, sub_graph_random_walk_sample, cls_sub_graph_extractor
from codes.citation_graph_data import citation_graph_reconstruction, citation_khop_graph_reconstruction
from codes.graph_pretrained_dataset import SubGraphPairDataset
from codes.ogb_graph_data import ogb_khop_graph_reconstruction
from codes.graph_train_dataset import NodePredSubGraphDataset, NodeSubGraphDataset
from codes.knowledge_graph_data import knowledge_graph_khop_reconstruction
from core.utils import seed_everything
from torch.utils.data import DataLoader
from numpy import random
seed_everything(seed=45)
##++++++++++++++
# kg_name = 'FB15k-237'
# fanouts = [15,10,5,5]
# graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
#     knowledge_graph_khop_reconstruction(dataset=kg_name, hop_num=5)
# print((graph.in_degrees() == 0).sum())
# start_time = time()
# kg_dataset = SubGraphPairDataset(graph=graph, nentity=number_of_nodes, nrelation=number_of_relations,
#                                    special_entity2id=special_entity_dict,
#                                    special_relation2id=special_relation_dict,
#                                    fanouts=fanouts)
# kg_dataloader = DataLoader(dataset=kg_dataset,
#                                  batch_size=16,
#                                  collate_fn=SubGraphPairDataset.collate_fn)
# # for _ in tqdm(range(kg_dataset.len)):
# #     kg_dataset.__getitem__(_)
# for batch_idx, batch in tqdm(enumerate(kg_dataloader)):
#     batch_graph, batch_cls = batch['batch_graph_1']
#     print(batch_graph.ndata['nid'][batch_cls], batch_graph.number_of_nodes())
# print('Run time = {:.4f}'.format(time() - start_time))
# ##++++++++++++++
citation_data_name = 'citeseer'
graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, \
special_relation_dict, n_classes, n_feats = \
    citation_khop_graph_reconstruction(dataset=citation_data_name, hop_num=6)
print(graph.ndata)
print('Number of nodes with 0 in-degree = {}'.format((graph.in_degrees() == 0).sum()))
print(graph.number_of_nodes())
print(graph.ndata['label'])
print(graph.ndata)
start_time = time()
fanouts = [10,5,5,5]
# fanouts = [-1,-1,-1,-1]
# citation_dataset = SubGraphPairDataset(graph=graph, nentity=number_of_nodes,
#                                    nrelation=number_of_relations,
#                                    special_entity2id=special_entity_dict,
#                                    special_relation2id=special_relation_dict,
#                                    fanouts=fanouts)
# citation_dataloader = DataLoader(dataset=citation_dataset,
#                                  batch_size=16,
#                                  collate_fn=SubGraphPairDataset.collate_fn)
# for _ in tqdm(range(citation_dataset.len)):
#     citation_dataset.__getitem__(_)
#     break

citation_dataset = NodePredSubGraphDataset(graph=graph, nentity=number_of_nodes,
                                   nrelation=number_of_relations,
                                   special_entity2id=special_entity_dict,
                                   special_relation2id=special_relation_dict,
                                   fanouts=fanouts, graph_type='citation', data_type='train')
citation_dataloader = DataLoader(dataset=citation_dataset,
                                 batch_size=16,
                                 collate_fn=SubGraphPairDataset.collate_fn)
for _ in tqdm(range(citation_dataset.len)):
    citation_dataset.__getitem__(_)


# # print('Run time = {:.4f}'.format(time() - start_time))
# for batch_idx, batch in tqdm(enumerate(citation_dataloader)):
#     batch_graph, batch_cls = batch['batch_graph_2']
#     # print(batch_graph.ndata['nid'][batch_cls])
# print('Run time = {:.4f}'.format(time() - start_time))

# ogb_dataname = 'ogbn-arxiv'
#
# data = DglNodePropPredDataset(name=ogb_dataname, root=HOME_DATA_FOLDER)
# graph, labels = data[0]
# print(graph.number_of_nodes())
# print(labels.shape)
# graph, node_split_idx, node_features, number_of_nodes, number_of_relations, special_entity_dict,\
# special_relation_dict, n_classes, n_feats = \
#         ogb_khop_graph_reconstruction(dataset=ogb_dataname, hop_num=6)
# print('Number of nodes = {}'.format(number_of_nodes))
# print('Number of eges = {}'.format(graph.number_of_edges()))
# print('Node features = {}'.format(n_feats))
# print('Number of relations = {}'.format(number_of_relations))
# print('Number of nodes with 0 in-degree = {}'.format((graph.in_degrees() == 0).sum()))
# fanouts = [10, 5, 5]
#
# print(graph.number_of_edges())
#
#
# ogb_dataset = SubGraphPairDataset(graph=graph, nentity=number_of_nodes, nrelation=number_of_relations,
#                                   special_entity2id=special_entity_dict,
#                                   special_relation2id=special_relation_dict, fanouts=fanouts)
# #
# for _ in tqdm(range(ogb_dataset.len)):
#     ogb_dataset.__getitem__(_)


# def logbook_solution(logbook: list):
#     char_dict = {}
#     for ch_1, ch_2 in logbook:
#         start_i = ord(ch_1.upper())
#         end_i = ord(ch_2.upper())
#         if start_i > end_i:
#             temp = start_i
#             start_i = end_i
#             end_i = temp
#         if start_i not in char_dict:
#             char_dict[start_i] = end_i
#         else:
#             if end_i > char_dict[start_i]:
#                 char_dict[start_i] = end_i
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++
#     # dictionary will reduce the number of possible pairs in logbook
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++
#     sorted_keys = sorted(char_dict)
#     prev_start = sorted_keys[0]
#     prev_right = char_dict[prev_start]
#     max_len = prev_right - prev_start
#     for i in range(1, len(sorted_keys)):
#         if sorted_keys[i] <= prev_right:
#             prev_right = char_dict[sorted_keys[i]]
#         else:
#             prev_start = sorted_keys[i]
#             prev_right = char_dict[sorted_keys[i]]
#         cur_len = prev_right - prev_start
#         if max_len <= cur_len:
#             max_len = cur_len
#     return max_len
#
#
# def solution(logbook):
#     global_max = 0
#     sorted_log = []
#     for log in logbook:
#         if log[0] > log[1]:
#             sorted_log.append(log[1] + log[0])
#         else:
#             sorted_log.append(log)
#
#     logbook = sorted(sorted_log)
#     prev_right = logbook[0][1]
#     prev_start = logbook[0][0]
#     for log in logbook:
#         if log[0] <= prev_right:
#             cur_len = max(ord(log[1]), ord(prev_right)) - ord(prev_start)
#             global_max = max(global_max, cur_len)
#             prev_right = max(prev_right, log[1])
#         else:
#             prev_start = log[0]
#             prev_right = log[1]
#             # ++++
#             cur_len = ord(prev_right) - ord(prev_start)
#             global_max = max(global_max, cur_len)
#             # ++++
#     return global_max
#
# # def sqrt(x):
# #     last_guess = x/2.0
# #     while True:
# #         guess = (last_guess + x/last_guess)/2
# #         if abs(guess - last_guess) < .000001: # example threshold
# #             return guess
# #         last_guess = guess
# #
# #
# # def sqrt_solution(predicted, observed):
# #     # write your code in Python 3.6
# #     n = len(predicted)
# #     sum_se = 0
# #     for i in range(n):
# #         sum_se += (predicted[i] - observed[i]) ** 2
# #
# #     avg_se = sum_se / n
# #     print(avg_se)
# #
# #     def square(n, left, right):
# #         mid = (left + right) / 2
# #         square_sum = mid * mid
# #         if (square_sum == n) or (abs(square_sum - n) < 0.000001):
# #             return mid
# #         elif (square_sum < n):
# #             return square(n, mid, right)
# #         else:
# #             return square(n, left, mid)
# #
# #     def find_square_root(n):
# #         left = 1
# #         Ind = False
# #         while (Ind == False):
# #             if (left * left == n):
# #                 res = left
# #                 Ind = True
# #             elif (left * left > n):
# #                 res = square(n, left - 1, left)
# #                 Ind = True
# #             left += 1
# #         return res
# #     rmse = find_square_root(avg_se)
# #     return rmse
# #
# # def sqrt_solution(predicted, observed):
# #     # write your code in Python 3.6
# #     n = len(predicted)
# #     sum_se = 0
# #     for i in range(n):
# #         sum_se += (predicted[i] - observed[i]) ** 2
# #
# #     avg_se = sum_se / n
# #
# #     def find_square_root(x):
# #         last_guess = x / 2.0
# #         while True:
# #             guess = (last_guess + x / last_guess) / 2
# #             if abs(guess - last_guess) < .000001:  # example threshold
# #                 return guess
# #             last_guess = guess
# #     rmse = find_square_root(avg_se)
# #     return rmse
#
# if __name__ == '__main__':
#     print()
#     # x = ['BG', 'CA', 'FI', 'OK']
#     # x = ['BG', 'CA']
#     # x = ['FI', 'OK']
#     x = ['AB', 'AC', 'AD', 'OK']
#     # # x = ['AB', 'CF']
#     # # x = ['AB', 'AC', 'CF']
#     # # x = ['ab', 'aF']
#     y = logbook_solution(x)
#     print(y)
#     y = solution(x)
#     print(y)
#     # y = sqrt(3)
#     # print(y)
#
#     # x = [4, 25, 0.75, 11]
#     # y = [3, 21, -1.25, 13]
#     #
#     # z = sqrt_solution(x, y)
#     # print(z)
#
#
