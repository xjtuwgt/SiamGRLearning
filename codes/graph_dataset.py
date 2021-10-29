from dgl import DGLHeteroGraph
import torch
from torch.utils.data import Dataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from core.graph_utils import add_relation_ids_to_graph, construct_special_graph_dictionary

def citation_graph_reconstruction(dataset: str):
    if dataset == 'cora':
        data = CoraGraphDataset()
    elif dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    graph = data[0]
    node_features = graph.ndata.pop('feat')
    number_of_edges = graph.number_of_edges()
    edge_type_ids = torch.zeros(number_of_edges, dtype=torch.long)
    graph = add_relation_ids_to_graph(graph=graph, edge_type_ids=edge_type_ids)
    nentities, nrealtions = graph.number_of_nodes(), 1
    return graph, node_features, nentities, nrealtions

def citation_khop_graph_reconstruction(dataset: str, hop_num=5):
    graph, node_features, nentities, nrealtions = citation_graph_reconstruction(dataset=dataset)
    graph, number_of_nodes, number_of_relations, \
    special_entity_dict, special_relation_dict = construct_special_graph_dictionary(graph=graph, n_entities=nentities,
                                       n_relations=nrealtions, hop_num=hop_num)
    number_of_added_nodes = number_of_nodes - nentities
    print('Added number of nodes = {}'.format(number_of_added_nodes))
    assert len(special_entity_dict) == number_of_added_nodes
    if number_of_added_nodes > 0:
        added_node_features = torch.zeros((number_of_added_nodes, node_features.shape[1]))
        node_features = torch.cat([node_features, added_node_features], dim=0)
    return graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict


class CitationSubGraphDataset(Dataset):
    def __init__(self, g: DGLHeteroGraph, nentity: int, nrelation: int,
                 fanouts: list, special_entity2id: dict, special_relation2id: dict,
                 reverse=False, edge_dir='in'):
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
        self.special_entity2id = special_entity2id
        self.special_relation2id = special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        anchor_node_ids = torch.LongTensor([idx])
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])

        # if self.edge_dir == 'in':
        #     in_neighbors_dict, in_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids,
        #                                                        cls_node_ids=cls_node_ids,
        #                                                        g=self.g, fanouts=self.fanouts, edge_dir=self.edge_dir)
        #     out_neighbors_dict, out_edge_dict = None, None
        # elif self.edge_dir == 'out':
        #     in_neighbors_dict, in_edge_dict = None, None
        #     out_neighbors_dict, out_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids,
        #                                                          cls_node_ids=cls_node_ids,
        #                                                          g=self.g, fanouts=self.fanouts, edge_dir=self.edge_dir)
        # elif self.edge_dir == 'all':
        #     in_neighbors_dict, in_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids,
        #                                                        cls_node_ids=cls_node_ids,
        #                                                        g=self.g, fanouts=self.fanouts, edge_dir='in')
        #     out_neighbors_dict, out_edge_dict = directed_sub_graph(anchor_node_ids=anchor_node_ids,
        #                                                          cls_node_ids=cls_node_ids,
        #                                                          g=self.g, fanouts=self.fanouts, edge_dir='out')
        # else:
        #     raise 'Edge direction {} is not supported'.format(self.edge_dir)
