from core.graph_utils import directed_sub_graph
from dgl import DGLHeteroGraph
import torch
import dgl
from torch.utils.data import Dataset

class SubGraphDataset(Dataset):
    def __init__(self, g: DGLHeteroGraph, nentity: int, nrelation: int,
                 fanouts: list, special_entity2id: dict, special_relation2id: dict,
                 replace=False, reverse=False, edge_dir='in'):
        assert len(fanouts) > 0
        self.fanouts = fanouts
        self.hop_num = len(fanouts)
        self.g = g
        #####################
        if len(special_entity2id) > 0:
            self.len = g.number_of_nodes() - len(special_entity2id) ## no need to extract sub-graph of special entities
        else:
            self.len = g.number_of_nodes()
        #####################
        self.nentity = nentity
        self.nrelation = nrelation
        self.reverse = reverse
        self.fanouts = fanouts ## list of int == number of hops for sampling
        self.edge_dir = edge_dir ## "in", "out", "all"
        self.replace = replace
        self.special_entity2id = special_entity2id
        self.special_relation2id = special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        anchor_node_ids = torch.LongTensor([idx])
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        if self.edge_dir == 'in':
            in_neighbors_dict, in_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids,
                                                               cls_node_ids=cls_node_ids,
                                                               g=self.g, fanouts=self.fanouts, edge_dir=self.edge_dir)
            out_neighbors_dict, out_edge_dict = None, None
        elif self.edge_dir == 'out':
            in_neighbors_dict, in_edge_dict = None, None
            out_neighbors_dict, out_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids,
                                                                 cls_node_ids=cls_node_ids,
                                                                 g=self.g, fanouts=self.fanouts, edge_dir=self.edge_dir)
        elif self.edge_dir == 'all':
            in_neighbors_dict, in_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids,
                                                               cls_node_ids=cls_node_ids,
                                                               g=self.g, fanouts=self.fanouts, edge_dir='in')
            out_neighbors_dict, out_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids,
                                                                 cls_node_ids=cls_node_ids,
                                                                 g=self.g, fanouts=self.fanouts, edge_dir='out')
        else:
            raise 'Edge direction {} is not supported'.format(self.edge_dir)
