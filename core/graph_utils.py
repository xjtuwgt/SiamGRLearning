import dgl
import numpy as np
import torch
from dgl.sampling import sample_neighbors
from dgl.sampling.randomwalks import random_walk
from torch import Tensor
from time import time

def construct_special_graph_dictionary(graph, hop_num: int, n_relations: int, n_entities: int):
    special_entity_dict = {}
    special_relation_dict = {}
    number_nodes = n_entities
    assert number_nodes == graph.number_of_nodes()
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    special_entity_dict['cls'] = number_nodes ## for graph-level representation learning
    special_entity_dict['mask'] = number_nodes + 1 ## for node mask
    graph.add_nodes(2) ### add such 'cls' token as a new mask entity++++ adding two more nodes
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for hop in range(hop_num):
        special_relation_dict['in_hop_{}_r'.format(hop + 1)] = n_relations + (2 * hop)
        special_relation_dict['out_hop_{}_r'.format(hop + 1)] = n_relations + (2 * hop + 1)
    n_relations = n_relations + 2 * hop_num
    special_relation_dict['cls_r'] = n_relations ## connect each node to cls token;
    n_relations = n_relations + 1
    special_relation_dict['loop_r'] = n_relations ## self-loop relation
    n_relations = n_relations + 1
    special_relation_dict['mask_r'] = n_relations ### for edge mask
    number_of_nodes = graph.number_of_nodes()
    number_of_relations = n_relations
    return graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict

def graph_to_bidirected(graph, number_of_relations: int=None, is_hete_graph=True):
    assert 'tid' in graph.edata.keys()
    src_nodes, dst_nodes, edge_ids = graph.edges(form='all')
    edge_tids = graph.edata['tid'][edge_ids]
    if is_hete_graph:
        assert number_of_relations is not None
        rev_edge_tids = edge_tids + number_of_relations
        graph.add_edges(dst_nodes, src_nodes, {'tid': rev_edge_tids})
    else:
        graph.add_edges(dst_nodes, src_nodes, {'tid': edge_tids})
    return graph

def add_relation_ids_to_graph(graph, edge_type_ids: Tensor):
    graph.edata['tid'] = edge_type_ids
    return graph

def sub_graph_neighbor_sample_unique(graph, anchor_node_ids: Tensor, cls_node_ids: Tensor, fanouts: list,
                     edge_dir: str = 'in', debug=False):
    """
    :param graph: dgl graph
    :param anchor_node_ids: LongTensor
    :param cls_node_ids: LongTensor
    :param fanouts: size = hop_number, (list, each element represents the number of sampling neighbors)
    :param edge_dir:  'in' or 'out'
    :return:
    """
    assert edge_dir in {'in', 'out'}
    start_time = time() if debug else 0
    neighbors_dict = {'anchor': (anchor_node_ids, torch.tensor([1], dtype=torch.long)),
                      'cls': (cls_node_ids, torch.tensor([1], dtype=torch.long))}
    edge_dict = {} ## sampled edge dictionary: (head, t_id, tail)
    hop, hop_number = 1, len(fanouts)
    while hop < hop_number + 1:
        if hop == 1:
            node_ids = neighbors_dict['anchor']
        else:
            node_ids = neighbors_dict['{}_hop_{}'.format(edge_dir, hop - 1)]
        sg = sample_neighbors(g=graph, nodes=node_ids[0], edge_dir=edge_dir, fanout=fanouts[hop - 1])
        sg_src, sg_dst = sg.edges()
        sg_eids, sg_tids = sg.edata[dgl.EID], sg.edata['tid']
        sg_src_list, sg_dst_list = sg_src.tolist(), sg_dst.tolist()
        sg_eid_list, sg_tid_list = sg_eids.tolist(), sg_tids.tolist()
        for _, eid in enumerate(sg_eid_list):
            edge_dict[eid] = (sg_src_list[_], sg_tid_list[_], sg_dst_list[_])
        hop_neighbor = sg_src if edge_dir == 'in' else sg_dst
        hop_neighbor, hop_neighbor_counts = torch.unique(hop_neighbor, return_counts=True)
        neighbors_dict['{}_hop_{}'.format(edge_dir, hop)] = (hop_neighbor, hop_neighbor_counts)
        hop = hop + 1
    end_time = time() if debug else 0
    if debug:
        print('Sampling time = {:.4f} seconds'.format(end_time - start_time))
    return neighbors_dict, edge_dict


def sub_graph_neighbor_sample(graph, anchor_node_ids: Tensor, cls_node_ids: Tensor, fanouts: list, edge_dir: str = 'in',
                              debug=False):
    """
    :param graph: dgl graph
    :param anchor_node_ids: LongTensor
    :param cls_node_ids: LongTensor
    :param fanouts: size = hop_number, (list, each element represents the number of sampling neighbors)
    :param edge_dir:  'in' or 'out'
    :return:
    """
    assert edge_dir in {'in', 'out'}
    start_time = time() if debug else 0
    neighbors_dict = {'anchor': anchor_node_ids, 'cls': cls_node_ids}
    edge_dict = {} ## sampled edge dictionary: (head, t_id, tail)
    hop, hop_number = 1, len(fanouts)
    while hop < hop_number + 1:
        if hop == 1:
            node_ids = neighbors_dict['anchor']
        else:
            node_ids = neighbors_dict['{}_hop_{}'.format(edge_dir, hop - 1)]
        sg = sample_neighbors(g=graph, nodes=node_ids, edge_dir=edge_dir, fanout=fanouts[hop - 1])
        sg_src, sg_dst = sg.edges()
        sg_eids, sg_tids = sg.edata[dgl.EID], sg.edata['tid']
        sg_src_list, sg_dst_list = sg_src.tolist(), sg_dst.tolist()
        sg_eid_list, sg_tid_list = sg_eids.tolist(), sg_tids.tolist()
        for _, eid in enumerate(sg_eid_list):
            edge_dict[eid] = (sg_src_list[_], sg_tid_list[_], sg_dst_list[_])
        hop_neighbor = sg_src if edge_dir == 'in' else sg_dst
        neighbors_dict['{}_hop_{}'.format(edge_dir, hop)] = hop_neighbor
        hop = hop + 1
    end_time = time() if debug else 0
    if debug:
        print('Sampling time = {:.4f} seconds'.format(end_time - start_time))
    neighbors_dict = dict([(k, torch.unique(v, return_counts=True))for k, v in neighbors_dict.items()])
    return neighbors_dict, edge_dict

def sub_graph_random_walk_sample(graph, anchor_node_ids: Tensor, cls_node_ids: Tensor, fanouts: list,
                     edge_dir: str = 'in', debug=False):
    """
    :param graph:
    :param anchor_node_ids:
    :param cls_node_ids:
    :param hop_num:
    :param edge_dir:
    :param debug:
    :return:
    """
    assert edge_dir in {'in', 'out'}
    start_time = time() if debug else 0
    if edge_dir == 'in':
        raw_graph = dgl.reverse(graph, copy_ndata=True, copy_edata=True)
    else:
        raw_graph = graph
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    walk_length = len(fanouts) + 1
    num_traces = torch.prod(torch.tensor(fanouts, dtype=torch.long))
    assert num_traces > 1
    neighbors_dict = {'anchor': (anchor_node_ids, torch.tensor([1], dtype=torch.long))}
    neighbors_dict['cls'] = (cls_node_ids, torch.tensor([1], dtype=torch.long))
    edge_dict = {} ## sampled edge dictionary: (head, t_id, tail)
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    anchor_node_ids = anchor_node_ids.repeat(num_traces)
    traces, types = random_walk(g=raw_graph, nodes=anchor_node_ids, length=walk_length)
    for hop in range(1, walk_length):
        neighbors_dict['{}_hop_{}'.format(edge_dir, hop)] = torch.unique(traces[:,hop], return_counts=True)
    src_nodes, dst_nodes = traces[:,:-1].flatten(), traces[:,1:].flatten()
    number_of_nodes = graph.number_of_nodes()
    edge_ids = src_nodes * number_of_nodes + dst_nodes
    edge_ids[dst_nodes == -1] = -1
    unique_edge_id = torch.unique(edge_ids[edge_ids >= 0])
    src_nodes, dst_nodes = unique_edge_id // number_of_nodes, unique_edge_id % number_of_nodes
    src_node_list, dst_node_list = src_nodes.tolist(), dst_nodes.tolist()
    edge_ids = raw_graph.edge_ids(src_nodes, dst_nodes)
    edge_tids = raw_graph.edata['tid'][edge_ids]
    eid_list, tid_list = edge_ids.tolist(), edge_tids.tolist()
    for _, eid in enumerate(eid_list):
        edge_dict[eid] = (src_node_list[_], tid_list[_], dst_node_list[_])
    end_time = time() if debug else 0
    if debug:
        print('Sampling time = {:.4f} seconds'.format(end_time - start_time))
    return neighbors_dict, edge_dict

def sub_graph_extractor(graph, edge_dict: dict, neighbors_dict: dict, bi_directed:bool = True):
    if len(edge_dict) == 0:
        return single_node_graph_extractor(graph=graph, neighbors_dict=neighbors_dict)
    edge_ids = list(edge_dict.keys())
    if bi_directed:
        parent_triples = np.array(list(edge_dict.values()))
        rev_edge_ids = graph.edge_ids(parent_triples[:,2], parent_triples[:,0]).tolist()
        rev_edge_ids = [_ for _ in rev_edge_ids if _ not in edge_dict]
        rev_edge_ids = sorted(set(rev_edge_ids), key=rev_edge_ids.index)
    else:
        rev_edge_ids = []
    edge_ids = edge_ids + rev_edge_ids
    subgraph = graph.edge_subgraph(edges=edge_ids)
    return subgraph

def single_node_graph_extractor(graph, neighbors_dict: dict):
    anchor_ids = neighbors_dict['anchor'][0]
    sub_graph = graph.subgraph(anchor_ids)
    return sub_graph

def cls_sub_graph_extractor(graph, edge_dict: dict, neighbors_dict: dict, special_relation_dict: dict,
                            bi_directed: bool = True, debug=False):
    start_time = time() if debug else 0
    subgraph = sub_graph_extractor(graph=graph, edge_dict=edge_dict, bi_directed=bi_directed,
                                   neighbors_dict=neighbors_dict)
    subgraph.add_nodes(1)
    cls_parent_node_id = neighbors_dict['cls'][0][0].data.item()
    subgraph.ndata['nid'][-1] = cls_parent_node_id
    parent_node_ids, sub_node_ids = subgraph.ndata['nid'].tolist(), subgraph.nodes().tolist()
    parent2sub_dict = dict(zip(parent_node_ids, sub_node_ids))
    cls_idx = parent2sub_dict[cls_parent_node_id]
    assert cls_idx == subgraph.number_of_nodes() - 1
    cls_relation = [special_relation_dict['cls_r']] * (2 * (subgraph.number_of_nodes() - 1))
    cls_relation = torch.tensor(cls_relation, dtype=torch.long)
    cls_src_nodes = [cls_idx] * (subgraph.number_of_nodes() - 1)
    cls_src_nodes = torch.tensor(cls_src_nodes, dtype=torch.long)
    cls_dst_nodes = torch.arange(0, subgraph.number_of_nodes()-1)
    cls_src, cls_dst = torch.cat((cls_src_nodes, cls_dst_nodes)), np.concatenate((cls_dst_nodes, cls_src_nodes))
    subgraph.add_edges(cls_src, cls_dst, {'tid': cls_relation})
    end_time = time() if debug else 0
    if debug:
        print('CLS sub-graph construction time = {:.4f} seconds'.format(end_time - start_time))
    return subgraph, parent2sub_dict

def cls_sub_graph_augmentation(subgraph, parent2sub_dict: dict, neighbors_dict: dict,
                               special_relation_dict: dict, bi_directed: bool = True):
    anch_parent_node_id = neighbors_dict['anchor'][0][0].data.item()
    anch_idx = parent2sub_dict[anch_parent_node_id]
    assert anch_idx < subgraph.number_of_nodes() - 1
