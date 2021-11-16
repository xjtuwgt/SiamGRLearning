from dgl import DGLHeteroGraph
import torch
import dgl
from numpy import random
from torch.utils.data import Dataset
from core.graph_utils import sub_graph_neighbor_sample, cls_sub_graph_extractor, \
    cls_anchor_sub_graph_augmentation

class SubGraphPairDataset(Dataset):
    def __init__(self, graph: DGLHeteroGraph, nentity: int, nrelation: int, fanouts: list,
                 special_entity2id: dict, special_relation2id: dict, bi_directed=True, edge_dir='in'):
        assert len(fanouts) > 0
        self.fanouts = fanouts ## list of int == number of hops for sampling
        self.hop_num = len(fanouts)
        self.g = graph
        #####################
        if len(special_entity2id) > 0:
            self.len = graph.number_of_nodes() - len(special_entity2id) ## no need to extract sub-graph of special entities
        else:
            self.len = graph.number_of_nodes()
        #####################
        self.nentity, self.nrelation = nentity, nrelation
        self.bi_directed = bi_directed
        self.edge_dir = edge_dir ## "in", "out"
        self.special_entity2id, self.special_relation2id = special_entity2id, special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        anchor_node_ids = torch.LongTensor([idx])
        samp_hop_num = random.randint(2, self.hop_num+1)
        samp_fanouts = self.fanouts[:samp_hop_num]
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        neighbors_dict, node_arw_label_dict, edge_dict = sub_graph_neighbor_sample(graph=self.g, anchor_node_ids=anchor_node_ids,
                                                             cls_node_ids=cls_node_ids, fanouts=samp_fanouts,
                                                             edge_dir=self.edge_dir, debug=False)
        subgraph, parent2sub_dict = cls_sub_graph_extractor(graph=self.g, edge_dict=edge_dict,
                                                            neighbors_dict=neighbors_dict,
                                                            special_relation_dict=self.special_relation2id,
                                                            node_arw_label_dict=node_arw_label_dict,
                                                            bi_directed=self.bi_directed, debug=False)

        aug_subgraph = cls_anchor_sub_graph_augmentation(subgraph=subgraph, parent2sub_dict=parent2sub_dict,
                                          neighbors_dict=neighbors_dict, edge_dir=self.edge_dir,
                                          special_relation_dict=self.special_relation2id)
        return subgraph, aug_subgraph

    @staticmethod
    def collate_fn(data):
        assert len(data[0]) == 2
        batch_graphs_1 = dgl.batch([_[0] for _ in data])
        batch_graph_1_cls = torch.as_tensor([_[0].number_of_nodes() for _ in data], dtype=torch.long)
        batch_graph_1_cls = torch.cumsum(batch_graph_1_cls, dim=0) - 1

        batch_graph_2_cls = torch.as_tensor([_[1].number_of_nodes() for _ in data], dtype=torch.long)
        batch_graph_2_cls = torch.cumsum(batch_graph_2_cls, dim=0) - 1
        batch_graphs_2 = dgl.batch([_[1] for _ in data])
        return {'batch_graph_1': (batch_graphs_1, batch_graph_1_cls),
                'batch_graph_2': (batch_graphs_2, batch_graph_2_cls)}

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SubGraphDataset(Dataset):
    def __init__(self, graph: DGLHeteroGraph, nentity: int, nrelation: int, fanouts: list,
                 special_entity2id: dict, special_relation2id: dict, bi_directed=True, edge_dir='in'):
        assert len(fanouts) > 0
        self.fanouts = fanouts ## list of int == number of hops for sampling
        self.hop_num = len(fanouts)
        self.g = graph
        #####################
        if len(special_entity2id) > 0:
            self.len = graph.number_of_nodes() - len(special_entity2id) ## no need to extract sub-graph of special entities
        else:
            self.len = graph.number_of_nodes()
        #####################
        self.nentity, self.nrelation = nentity, nrelation
        self.bi_directed = bi_directed
        self.edge_dir = edge_dir ## "in", "out"
        self.special_entity2id, self.special_relation2id = special_entity2id, special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        anchor_node_ids = torch.LongTensor([idx])
        samp_hop_num = random.randint(2, self.hop_num+1)
        samp_fanouts = self.fanouts[:samp_hop_num]
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        neighbors_dict, node_arw_label_dict, edge_dict = sub_graph_neighbor_sample(graph=self.g, anchor_node_ids=anchor_node_ids,
                                                             cls_node_ids=cls_node_ids, fanouts=samp_fanouts,
                                                             edge_dir=self.edge_dir, debug=False)
        subgraph, parent2sub_dict = cls_sub_graph_extractor(graph=self.g, edge_dict=edge_dict,
                                                            neighbors_dict=neighbors_dict,
                                                            special_relation_dict=self.special_relation2id,
                                                            node_arw_label_dict=node_arw_label_dict,
                                                            bi_directed=self.bi_directed, debug=False)
        return subgraph

    @staticmethod
    def collate_fn(data):
        batch_graph_cls = torch.as_tensor([_.number_of_nodes() for _ in data], dtype=torch.long)
        batch_graph_cls = torch.cumsum(batch_graph_cls, dim=0) - 1
        batch_graphs = dgl.batch([_ for _ in data])
        return {'batch_graph': (batch_graphs, batch_graph_cls)}