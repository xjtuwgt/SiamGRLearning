from dgl.sampling import sample_neighbors
from dgl import DGLHeteroGraph
from collections import OrderedDict
import itertools
import torch
import dgl
from torch.utils.data import Dataset

class SubGraphPairDataset(Dataset):
    def __init__(self, g: DGLHeteroGraph, nentity: int, nrelation: int,
                 fanouts: list, special_entity2id: dict, special_relation2id: dict,
                 replace=False, reverse=False, edge_dir='in'):
        assert len(fanouts) > 0
        self.fanouts = fanouts
        self.hop_num = len(fanouts)
        self.g = g
        #####################
        if len(special_entity2id) > 0:
            self.len = g.number_of_nodes() - len(special_entity2id)  ## no need to extract sub-graph of special entities
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
    #     if self.edge_dir == 'in':
    #         in_neighbors_dict, in_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids, cls_node_ids=cls_node_ids,
    #                                          g=self.g, fanouts=self.fanouts, edge_dir=self.edge_dir)
    #         out_neighbors_dict, out_edge_dict = None, None
    #     elif self.edge_dir == 'out':
    #         in_neighbors_dict, in_edge_dict = None, None
    #         out_neighbors_dict, out_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids, cls_node_ids=cls_node_ids,
    #                                            g=self.g, fanouts=self.fanouts, edge_dir=self.edge_dir)
    #     elif self.edge_dir == 'all':
    #         in_neighbors_dict, in_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids, cls_node_ids=cls_node_ids,
    #                                          g=self.g, fanouts=self.fanouts, edge_dir='in')
    #         out_neighbors_dict, out_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids, cls_node_ids=cls_node_ids,
    #                                            g=self.g, fanouts=self.fanouts, edge_dir='out')
    #     else:
    #         raise 'Edge direction {} is not supported'.format(self.edge_dir)
    #
    #     sub_graph, cls_sub_id = sub_graph_extractor(neighbor_dict_pair=(in_neighbors_dict, out_neighbors_dict),
    #                                                 edge_dict_pair=(in_edge_dict, out_edge_dict),
    #                                            cls_id=self.special_entity2id['cls'], special_relation2id=self.special_relation2id,
    #                                            edge_dir=self.edge_dir, reverse=self.reverse, n_relations=self.nrelation)
    #     dense_sub_graph, anchor_sub_id = dense_graph_constructor(neighbor_dict_pair=(in_neighbors_dict, out_neighbors_dict), sub_graph=sub_graph,
    #                                               hop_num=self.hop_num,
    #                                               special_relation2id=self.special_relation2id, reverse=self.reverse, edge_dir=self.edge_dir)
    #     return anchor_node_ids, cls_sub_id, anchor_sub_id, sub_graph, dense_sub_graph
    #
    # @staticmethod
    # def collate_fn(data):
    #     anchor_nodes = torch.cat([_[0] for _ in data], dim=0)
    #     cls_sub_ids = torch.cat([_[1] for _ in data], dim=0)
    #     anchor_sub_ids = torch.cat([_[2] for _ in data], dim=0)
    #     batch_graphs = dgl.batch(list(itertools.chain.from_iterable([(_[3], _[4]) for _ in data]))) ## batch size * 2
    #
    #     number_of_nodes = torch.LongTensor([sum([_[3].number_of_nodes() for _ in data])])[0]
    #     dense_number_of_nodes = torch.LongTensor([sum([_[4].number_of_nodes() for _ in data])])[0]
    #     assert number_of_nodes == dense_number_of_nodes
    #     sparse_number_of_edges = torch.LongTensor([sum([_[3].number_of_edges() for _ in data])])[0]
    #     dense_number_of_edges = sum([_[4].number_of_edges() for _ in data])
    #     edge_number = torch.LongTensor([sparse_number_of_edges + dense_number_of_edges])[0]
    #     batch = {'anchor': anchor_nodes, 'cls': cls_sub_ids, 'anchor_sub': anchor_sub_ids,
    #              'node_number': number_of_nodes, 'edge_number': edge_number, 'batch_graph': batch_graphs}
    #     return batch