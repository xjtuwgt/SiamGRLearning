from dgl import DGLHeteroGraph
import torch
from numpy import random
from torch.utils.data import Dataset
from core.graph_utils import sub_graph_neighbor_sample, cls_sub_graph_extractor

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
        print(subgraph.ndata)
        print(subgraph.edata)
        # print(subgraph.ndata['nid'])
        # print(subgraph.ndata['n_order'])
        # print(subgraph.number_of_nodes(), len(neighbor2pathlen_dict))