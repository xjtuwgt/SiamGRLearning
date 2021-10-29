from dgl.sampling import sample_neighbors
from torch import LongTensor
from collections import OrderedDict
import torch
import dgl
from copy import deepcopy
import dgl.backend as F
import numpy as np
import scipy as sp
def construct_special_graph_dictionary(graph, hop_num: int, n_relations: int, n_entities: int):
    special_entity_dict = {}
    special_relation_dict = {}
    number_nodes = n_entities
    assert number_nodes == graph.number_of_nodes()
    special_entity_dict['cls'] = number_nodes ## for graph-level representation learning
    special_entity_dict['mask'] = number_nodes + 1 ## for node mask
    graph.add_nodes(2) ### add such 'cls' token as a new mask entity++++ adding two more nodes
    for hop in range(hop_num):
        special_relation_dict['in_hop_{}'.format(hop + 1)] = n_relations + (2 * hop)
        special_relation_dict['out_hop_{}'.format(hop + 1)] = n_relations + (2 * hop + 1)
    n_relations = n_relations + 2 * hop_num
    special_relation_dict['cls_r'] = n_relations ## connect each node to cls token;
    n_relations = n_relations + 1
    special_relation_dict['loop_r'] = n_relations ## self-loop relation
    n_relations = n_entities + 1
    special_relation_dict['mask_r'] = n_relations ### for edge mask
    number_of_nodes = graph.number_of_nodes()
    number_of_relations = n_relations
    return graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict

def directed_sub_graph(anchor_node_ids: LongTensor, cls_node_ids: LongTensor, fanouts: list, graph, edge_dir: str = 'in'):
    """
    :param anchor_node_ids: LongTensor
    :param cls_node_ids: LongTensor
    :param fan-outs: size = hop_number, (list, each element represents the number of sampling neighbors)
    :param g: dgl graph
    :param edge_dir: 'in' or 'out'
    :return:
    """
    assert edge_dir in {'in', 'out'}
    neighbors_dict = {'anchor': anchor_node_ids}
    neighbors_dict['cls'] = cls_node_ids ## connected to all the other nodes for graph-level representation learning
    edge_dict = {} ## sampled edge dictionary: (head, t_id, tail)
    hop = 1
    hop_number = len(fanouts)
    while hop < hop_number + 1:
        if hop == 1:
            node_ids = neighbors_dict['anchor']
        else:
            node_ids = neighbors_dict['{}_hop_{}'.format(edge_dir, hop - 1)]
        sg = sample_neighbors(g=graph, nodes=node_ids, edge_dir=edge_dir, fanout=fanouts[hop - 1])
        sg_src, sg_dst = sg.edges()
        sg_eids, sg_tids = sg.edata['_ID'], sg.edata['tid']
        sg_src_list, sg_dst_list = sg_src.tolist(), sg_dst.tolist()
        sg_eid_list, sg_tid_list = sg_eids.tolist(), sg_tids.tolist()
        for eid, src_id, tid, dst_id in zip(sg_eid_list, sg_src_list, sg_tid_list, sg_dst_list):
            edge_dict[eid] = (src_id, tid, dst_id)
        if edge_dir == 'in':
            hop_neighbor = sg_src
        else:
            hop_neighbor = sg_dst
        neighbors_dict['{}_hop_{}'.format(edge_dir, hop)] = hop_neighbor
        hop = hop + 1
    return neighbors_dict, edge_dict