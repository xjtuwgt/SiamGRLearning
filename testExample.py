from core.layers import EmbeddingLayer
import torch
from tqdm import tqdm
import dgl
from core.kg_utils import KGDataset, knowledge_graph_construction_from_triples, kg_data_path_collection
from core.graph_utils import graph_to_bidirected
from core.graph_utils import sub_graph_neighbor_sample, sub_graph_neighbor_sample_unique, \
    sub_graph_random_walk_sample, cls_sub_graph_extractor
from codes.citation_graph_dataset import citation_graph_reconstruction, citation_khop_graph_reconstruction
from codes.graph_dataloader import SubGraphDataset
from codes.knowledge_graph_dataset import knowledge_graph_khop_reconstruction
from core.utils import seed_everything
seed_everything(seed=43)
fanouts = [10,5,3,2]

##++++++++++++++
# kg_name = 'FB15k-237'
# graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
#     knowledge_graph_khop_reconstruction(dataset=kg_name, hop_num=5)
# print((graph.in_degrees() == 0).sum())
# kg_dataset = SubGraphDataset(graph=graph, nentity=number_of_nodes, nrelation=number_of_relations,
#                                    special_entity2id=special_entity_dict,
#                                    special_relation2id=special_relation_dict,
#                                    fanouts=fanouts)
# for _ in tqdm(range(kg_dataset.len)):
#     kg_dataset.__getitem__(_)

##++++++++++++++
citation_data_name = 'cora'
graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
    citation_khop_graph_reconstruction(dataset=citation_data_name, hop_num=5)
print((graph.in_degrees() == 0).sum())
citation_dataset = SubGraphDataset(graph=graph, nentity=number_of_nodes,
                                   nrelation=number_of_relations,
                                   special_entity2id=special_entity_dict,
                                   special_relation2id=special_relation_dict,
                                   fanouts=fanouts)
for _ in tqdm(range(citation_dataset.len)):
    citation_dataset.__getitem__(_)