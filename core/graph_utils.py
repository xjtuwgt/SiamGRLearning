from dgl.sampling import sample_neighbors
from collections import OrderedDict
import torch
import dgl
from copy import deepcopy
import dgl.backend as F
import numpy as np
import scipy as sp

def directed_sub_graph(anchor_node_ids, cls_node_ids, fanouts: list, g, edge_dir: str = 'in'):
    """
    :param anchor_node_ids: list[LongTensor]
    :param cls_node_ids: list[LongTensor]
    :param fan-outs: size = hop_number, (list, each element represents the number of sampling neighbors)
    :param g: dgl graph
    :param edge_dir: 'in' or 'out'
    :return:
    """
    neighbors_dict = {'anchor': anchor_node_ids}
    neighbors_dict['cls'] = cls_node_ids ## connected to all the other nodes for graph-level representation learning
    edge_dict = {} ## sampled edge dictionary: (head, t_id, tail)
    hop = 1
    while hop < len(fanouts) + 1:
        if hop == 1:
            node_ids = neighbors_dict['anchor']
        else:
            node_ids = neighbors_dict['hop_{}'.format(hop - 1)]
        sg = sample_neighbors(g=g, nodes=node_ids, edge_dir=edge_dir, fanout=fanouts[hop - 1])
        sg_src, sg_dst = sg.edges()
        sg_eids, sg_tids = sg.edata['_ID'], sg.edata['tid']
        sg_src_list, sg_dst_list = sg_src.tolist(), sg_dst.tolist()
        sg_eid_list, sg_tid_list = sg_eids.tolist(), sg_tids.tolist()
        for eid, src_id, tid, dst_id in zip(sg_eid_list, sg_src_list, sg_tid_list, sg_dst_list):
            edge_dict[eid] = (src_id, tid, dst_id)
        if edge_dir == 'in':
            hop_neighbor = sg_src
        elif edge_dir == 'out':
            hop_neighbor = sg_dst
        else:
            raise 'Edge direction {} is not supported'.format(edge_dir)
        neighbors_dict['hop_{}'.format(hop)] = hop_neighbor
        hop = hop + 1
    return neighbors_dict, edge_dict