from dgl import DGLHeteroGraph
import torch
import dgl
from numpy import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from core.graph_utils import sub_graph_neighbor_sample, cls_sub_graph_extractor, cls_anchor_sub_graph_augmentation


class NodeSubGraphDataset(Dataset):
    """
    Graph representation learning without node labels
    """
    def __init__(self, graph: DGLHeteroGraph, nentity: int, nrelation: int, fanouts: list,
                 special_entity2id: dict, special_relation2id: dict, bi_directed=True, edge_dir='in'):
        assert len(fanouts) > 0
        self.fanouts = fanouts  # list of int == number of hops for sampling
        self.hop_num = len(fanouts)
        self.g = graph
        #####################
        if len(special_entity2id) > 0:
            self.len = graph.number_of_nodes() - len(special_entity2id)  # no sub-graph for special entities
        else:
            self.len = graph.number_of_nodes()
        #####################
        self.nentity, self.nrelation = nentity, nrelation
        self.bi_directed = bi_directed
        self.edge_dir = edge_dir  # "in", "out"
        self.special_entity2id, self.special_relation2id = special_entity2id, special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        anchor_node_ids = torch.LongTensor([idx])
        samp_hop_num = random.randint(2, self.hop_num+1)
        samp_fanouts = self.fanouts[:samp_hop_num]
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        neighbors_dict, node_arw_label_dict, edge_dict = \
            sub_graph_neighbor_sample(graph=self.g, anchor_node_ids=anchor_node_ids,
                                      cls_node_ids=cls_node_ids, fanouts=samp_fanouts,
                                      edge_dir=self.edge_dir, debug=False)
        subgraph, parent2sub_dict = cls_sub_graph_extractor(graph=self.g, edge_dict=edge_dict,
                                                            neighbors_dict=neighbors_dict,
                                                            special_relation_dict=self.special_relation2id,
                                                            node_arw_label_dict=node_arw_label_dict,
                                                            bi_directed=self.bi_directed, debug=False)
        sub_anchor_id = parent2sub_dict[idx]
        return subgraph, sub_anchor_id

    @staticmethod
    def collate_fn(data):
        assert len(data[0]) == 2
        batch_graph_cls = torch.as_tensor([_[0].number_of_nodes() for _ in data], dtype=torch.long)
        batch_graph_cls = torch.cumsum(batch_graph_cls, dim=0) - 1
        batch_graphs = dgl.batch([_[0] for _ in data])
        # ++++++++++++++++++++++++++++++++++++++++
        batch_anchor_id = torch.zeros(len(data), dtype=torch.long)
        for idx, _ in enumerate(data):
            if idx == 0:
                batch_anchor_id[idx] = _[1]
            else:
                batch_anchor_id[idx] = _[1] + batch_graph_cls[idx - 1].data.item() + 1
        # ++++++++++++++++++++++++++++++++++++++++
        return {'batch_graph': (batch_graphs, batch_graph_cls, batch_anchor_id)}


def citation_train_valid_test(graph, data_type: str):
    if data_type == 'train':
        data_mask = graph.ndata['train_mask']
    elif data_type == 'valid':
        data_mask = graph.ndata['val_mask']
    elif data_type == 'test':
        data_mask = graph.ndata['test_mask']
    else:
        raise 'Data type = {} is not supported'.format(data_type)
    data_len = data_mask.int().sum().item()
    data_node_ids = data_mask.nonzero().squeeze()
    return data_len, data_node_ids


def ogb_train_valid_test(node_split_idx: dict, data_type: str):
    data_node_ids = node_split_idx[data_type]
    data_len = data_node_ids.shape[0]
    return data_len, data_node_ids


class NodePredSubGraphDataset(Dataset):
    """
    Graph representation learning with node labels
    """
    def __init__(self, graph: DGLHeteroGraph, nentity: int, nrelation: int, fanouts: list,
                 special_entity2id: dict, special_relation2id: dict, data_type: str, graph_type: str,
                 bi_directed: bool = True, self_loop: bool = False, edge_dir: str = 'in',
                 node_split_idx: dict = None):
        assert len(fanouts) > 0 and (data_type in {'train', 'valid', 'test'})
        assert graph_type in {'citation', 'ogb'}
        self.fanouts = fanouts  # list of int == number of hops for sampling
        self.hop_num = len(fanouts)
        self.g = graph
        #####################
        if graph_type == 'ogb':
            assert node_split_idx is not None
            self.len, self.data_node_ids = ogb_train_valid_test(node_split_idx=node_split_idx, data_type=data_type)
        elif graph_type == 'citation':
            self.len, self.data_node_ids = citation_train_valid_test(graph=graph, data_type=data_type)
        else:
            raise 'Graph type = {} is not supported'.format(graph_type)
        assert self.len > 0
        #####################
        self.nentity, self.nrelation = nentity, nrelation
        self.bi_directed = bi_directed
        self.edge_dir = edge_dir  # "in", "out"
        self.self_loop = self_loop
        self.special_entity2id, self.special_relation2id = special_entity2id, special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        node_idx = self.data_node_ids[idx]
        anchor_node_ids = torch.LongTensor([node_idx])
        samp_hop_num = random.randint(2, self.hop_num+1)
        samp_fanouts = self.fanouts[:samp_hop_num]
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        neighbors_dict, node_arw_label_dict, edge_dict = \
            sub_graph_neighbor_sample(graph=self.g, anchor_node_ids=anchor_node_ids,
                                      cls_node_ids=cls_node_ids, fanouts=samp_fanouts,
                                      edge_dir=self.edge_dir, debug=False)
        subgraph, parent2sub_dict = cls_sub_graph_extractor(graph=self.g, edge_dict=edge_dict,
                                                            neighbors_dict=neighbors_dict,
                                                            special_relation_dict=self.special_relation2id,
                                                            node_arw_label_dict=node_arw_label_dict,
                                                            self_loop=self.self_loop,
                                                            bi_directed=self.bi_directed, debug=False)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # subgraph = cls_anchor_sub_graph_augmentation(subgraph=subgraph, parent2sub_dict=parent2sub_dict,
        #                                              neighbors_dict=neighbors_dict,
        #                                              special_relation_dict=self.special_relation2id,
        #                                              edge_dir=self.edge_dir, bi_directed=self.bi_directed)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sub_anchor_id = parent2sub_dict[node_idx.data.item()]
        class_label = self.g.ndata['label'][node_idx]
        return subgraph, class_label, sub_anchor_id

    @staticmethod
    def collate_fn(data):
        assert len(data[0]) == 3
        batch_graph_cls = torch.as_tensor([_[0].number_of_nodes() for _ in data], dtype=torch.long)
        batch_graph_cls = torch.cumsum(batch_graph_cls, dim=0) - 1
        batch_graphs = dgl.batch([_[0] for _ in data])
        batch_label = torch.as_tensor([_[1].data.item() for _ in data], dtype=torch.long)
        # ++++++++++++++++++++++++++++++++++++++++
        batch_anchor_id = torch.zeros(len(data), dtype=torch.long)
        for idx, _ in enumerate(data):
            if idx == 0:
                batch_anchor_id[idx] = _[2]
            else:
                batch_anchor_id[idx] = _[2] + batch_graph_cls[idx - 1].data.item() + 1
        # +++++++++++++++++++++++++++++++++++++++
        # 'cls' for graph level prediction and 'anchor' for node/edge level prediction
        return {'batch_graph': (batch_graphs, batch_graph_cls, batch_anchor_id), 'batch_label': batch_label}


class node_prediction_data_helper(object):
    def __init__(self, graph, fanouts, number_of_nodes: int, number_of_relations: int, num_class: int,
                 special_entity_dict: dict, special_relation_dict: dict, train_batch_size: int,
                 val_batch_size: int, graph_type: str, edge_dir: str = 'in', self_loop: bool = False,
                 node_split_idx: dict = None):
        self.graph = graph
        self.fanouts = fanouts
        self.number_of_nodes = number_of_nodes
        self.number_of_relations = number_of_relations
        self.special_entity_dict = special_entity_dict
        self.special_relation_dict = special_relation_dict
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_class = num_class
        self.graph_type = graph_type
        self.node_split_idx = node_split_idx
        self.self_loop = self_loop
        self.edge_dir = edge_dir

    def data_loader(self, data_type):
        dataset = NodePredSubGraphDataset(graph=self.graph, nentity=self.number_of_nodes,
                                          nrelation=self.number_of_relations,
                                          special_entity2id=self.special_entity_dict,
                                          special_relation2id=self.special_relation_dict,
                                          data_type=data_type, graph_type=self.graph_type,
                                          edge_dir=self.edge_dir, self_loop=self.self_loop,
                                          fanouts=self.fanouts, node_split_idx=self.node_split_idx)
        if data_type in {'train'}:
            batch_size = self.train_batch_size
            shuffle = True
        else:
            batch_size = self.val_batch_size
            shuffle = False
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                collate_fn=NodePredSubGraphDataset.collate_fn)
        return data_loader
