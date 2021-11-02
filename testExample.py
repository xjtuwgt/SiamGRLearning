from core.layers import EmbeddingLayer
import torch
from tqdm import tqdm
import dgl
from core.kg_utils import KGDataset, knowledge_graph_construction_from_triples, kg_data_path_collection
from core.graph_utils import graph_to_bidirected
from core.graph_utils import sub_graph_neighbor_sample, sub_graph_neighbor_sample_unique, \
    sub_graph_random_walk_sample, cls_sub_graph_extractor
from codes.citation_graph_dataset import citation_graph_reconstruction, citation_khop_graph_reconstruction



# from core.utils import seed_everything
# from codes.kg_dataset import knowledge_graph_khop_reconstruction
# seed_everything(seed=43)
# kg_name = 'FB15k-237'
# graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
#     knowledge_graph_khop_reconstruction(dataset=kg_name, hop_num=5)
# print((graph.in_degrees() == 0).sum())
# # print(number_of_relations, number_of_nodes)
# cls = torch.LongTensor([special_entity_dict['cls']])
# count = 0
# for node_id in tqdm(range(graph.number_of_nodes())):
#     anchor = torch.LongTensor([node_id])
#     x, y = sub_graph_neighbor_sample(graph=graph, anchor_node_ids=anchor, cls_node_ids=cls,
#                                fanouts=[10,5,5,5], edge_dir='out', debug=False)
#     if len(y) == 0:
#         count = count + 1
#         continue
#     # print(len(y))
#     # sub_graph = sub_graph_extractor(graph=graph, edge_dict=y)
#     x, y = cls_sub_graph_extractor(graph=graph, edge_dict=y, neighbors_dict=x, special_relation_dict=special_relation_dict,
#                             bi_directed=True, debug=False)
# #     # print(x)
# print(count)

##++++++++++++++
citation_data_name = 'citeseer'
graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
    citation_khop_graph_reconstruction(dataset=citation_data_name, hop_num=5)
print((graph.in_degrees() == 0).sum())
cls = torch.LongTensor([special_entity_dict['cls']])
count = 0
for node_id in tqdm(range(graph.number_of_nodes())):
    anchor = torch.LongTensor([node_id])
    x, y = sub_graph_neighbor_sample(graph=graph, anchor_node_ids=anchor, cls_node_ids=cls,
                               fanouts=[10,5,5,5,5], edge_dir='in', debug=False)
    if len(y) == 0:
        count = count + 1
        continue
    x, y = cls_sub_graph_extractor(graph=graph, edge_dict=y, neighbors_dict=x, special_relation_dict=special_relation_dict,
                            bi_directed=True, debug=False)
#     # print(x)
print(count)