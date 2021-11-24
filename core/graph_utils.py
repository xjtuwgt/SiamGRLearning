import dgl
import numpy as np
import torch
from numpy import random
from dgl.sampling import sample_neighbors
from dgl.sampling.randomwalks import random_walk
from torch import Tensor
from time import time
import copy


def construct_special_graph_dictionary(graph, hop_num: int, n_relations: int, n_entities: int):
    """
    :param graph:
    :param hop_num: number of hops to generate special relations
    :param n_relations: number of relations in graph
    :param n_entities: number of entities (nodes) in graph
    :return:
    """
    special_entity_dict = {}
    special_relation_dict = {}
    number_nodes = n_entities
    assert number_nodes == graph.number_of_nodes()
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    special_entity_dict['cls'] = number_nodes  # for graph-level representation learning
    special_entity_dict['mask'] = number_nodes + 1  # for node mask
    graph.add_nodes(2)  # add 'cls' node and 'mask' node
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for hop in range(hop_num):
        special_relation_dict['in_hop_{}_r'.format(hop + 1)] = n_relations + (2 * hop)
        special_relation_dict['out_hop_{}_r'.format(hop + 1)] = n_relations + (2 * hop + 1)
    n_relations = n_relations + 2 * hop_num
    special_relation_dict['cls_r'] = n_relations  # connect each node to cls token;
    n_relations = n_relations + 1
    special_relation_dict['loop_r'] = n_relations  # self-loop relation
    n_relations = n_relations + 1
    special_relation_dict['mask_r'] = n_relations  # for edge mask
    number_of_nodes = graph.number_of_nodes()
    number_of_relations = n_relations
    return graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def add_relation_ids_to_graph(graph, edge_type_ids: Tensor):
    """
    :param graph:
    :param edge_type_ids: add 'rid' to graph edge data --> type id
    :return:
    """
    graph.edata['rid'] = edge_type_ids
    return graph


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def sub_graph_neighbor_sample(graph, anchor_node_ids: Tensor, cls_node_ids: Tensor, fanouts: list, edge_dir: str = 'in',
                              debug=False):
    """
    :param graph: dgl graph
    :param anchor_node_ids: LongTensor
    :param cls_node_ids: LongTensor
    :param fanouts: size = hop_number, (list, each element represents the number of sampling neighbors)
    :param edge_dir:  'in' or 'out'
    :param debug:
    :return:
    """
    assert edge_dir in {'in', 'out'}
    start_time = time() if debug else 0
    neighbors_dict = {'anchor': anchor_node_ids, 'cls': cls_node_ids}
    edge_dict = {}  # sampled edge dictionary: (head, t_id, tail)
    hop, hop_number = 1, len(fanouts)
    while hop < hop_number + 1:
        if hop == 1:
            node_ids = neighbors_dict['anchor']
        else:
            node_ids = neighbors_dict['{}_hop_{}'.format(edge_dir, hop - 1)]
        sg = sample_neighbors(g=graph, nodes=node_ids, edge_dir=edge_dir, fanout=fanouts[hop - 1])
        sg_src, sg_dst = sg.edges()
        sg_eids, sg_tids = sg.edata[dgl.EID], sg.edata['rid']
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
    neighbors_dict = dict([(k, torch.unique(v, return_counts=True)) for k, v in neighbors_dict.items()])
    # #############################################################################################
    node_arw_label_dict = {anchor_node_ids[0].data.item(): 1, cls_node_ids[0].data.item(): 0}
    # key == parent node id, value = length/hop based label
    # ###########################################anonymous rand walk node labels###################
    for hop in range(1, hop_number + 1):
        hop_neighbors = neighbors_dict['{}_hop_{}'.format(edge_dir, hop)]
        for neighbor in hop_neighbors[0].tolist():
            if neighbor not in node_arw_label_dict:
                node_arw_label_dict[neighbor] = hop + 1
    ##############################################################################################
    return neighbors_dict, node_arw_label_dict, edge_dict


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def sub_graph_random_walk_sample(graph, anchor_node_ids: Tensor, cls_node_ids: Tensor, fanouts: list,
                                 edge_dir: str = 'in', debug=False):
    """
    :param graph:
    :param anchor_node_ids:
    :param cls_node_ids:
    :param fanouts:
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
    # ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    walk_length = len(fanouts) + 1
    num_traces = torch.prod(torch.tensor(fanouts, dtype=torch.long)).data.item()
    assert num_traces > 1
    neighbors_dict = {'anchor': (anchor_node_ids, torch.tensor([1], dtype=torch.long)),
                      'cls': (cls_node_ids, torch.tensor([1], dtype=torch.long))}
    edge_dict = {}  # sampled edge dictionary: (head, t_id, tail)
    # ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    anchor_node_ids = anchor_node_ids.repeat(num_traces)
    traces, types = random_walk(g=raw_graph, nodes=anchor_node_ids, length=walk_length)
    for hop in range(1, walk_length):
        neighbors_dict['{}_hop_{}'.format(edge_dir, hop)] = torch.unique(traces[:, hop], return_counts=True)
    src_nodes, dst_nodes = traces[:, :-1].flatten(), traces[:, 1:].flatten()
    number_of_nodes = graph.number_of_nodes()
    edge_ids = src_nodes * number_of_nodes + dst_nodes
    edge_ids[dst_nodes == -1] = -1
    unique_edge_id = torch.unique(edge_ids[edge_ids >= 0])
    src_nodes, dst_nodes = unique_edge_id // number_of_nodes, unique_edge_id % number_of_nodes
    src_node_list, dst_node_list = src_nodes.tolist(), dst_nodes.tolist()
    edge_ids = raw_graph.edge_ids(src_nodes, dst_nodes)
    edge_tids = raw_graph.edata['rid'][edge_ids]
    eid_list, tid_list = edge_ids.tolist(), edge_tids.tolist()
    for _, eid in enumerate(eid_list):
        edge_dict[eid] = (src_node_list[_], tid_list[_], dst_node_list[_])
    ##############################################################################################
    node_arw_label_dict = {anchor_node_ids[0].data.item(): 1, cls_node_ids[0].data.item(): 0}
    for hop in range(1, walk_length):
        hop_neighbors = neighbors_dict['{}_hop_{}'.format(edge_dir, hop)]
        for neighbor in hop_neighbors[0].tolist():
            if neighbor not in node_arw_label_dict:
                node_arw_label_dict[neighbor] = hop + 1
    ##############################################################################################
    end_time = time() if debug else 0
    if debug:
        print('Sampling time = {:.4f} seconds'.format(end_time - start_time))
    return neighbors_dict, node_arw_label_dict, edge_dict


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def sub_graph_extractor(graph, edge_dict: dict, neighbors_dict: dict, bi_directed: bool = True):
    """
    :param graph: original graph
    :param edge_dict: edge dictionary: eid--> (src_node, edge_type, dst_node)
    :param neighbors_dict: {cls, anchor, hop} -> ((neighbors, neighbor counts))
    :param bi_directed: whether get bi-directional graph
    :return:
    """
    if len(edge_dict) == 0:
        assert 'anchor' in neighbors_dict
        return single_node_graph_extractor(graph=graph, neighbors_dict=neighbors_dict)
    edge_ids = list(edge_dict.keys())
    if bi_directed:
        parent_triples = np.array(list(edge_dict.values()))
        rev_edge_ids = graph.edge_ids(parent_triples[:, 2], parent_triples[:, 0]).tolist()
        rev_edge_ids = [_ for _ in rev_edge_ids if _ not in edge_dict]  # adding new edges as graph is bi_directed
        rev_edge_ids = sorted(set(rev_edge_ids), key=rev_edge_ids.index)
    else:
        rev_edge_ids = []
    edge_ids = edge_ids + rev_edge_ids
    subgraph = graph.edge_subgraph(edges=edge_ids)
    return subgraph


def single_node_graph_extractor(graph, neighbors_dict: dict):
    """
    :param graph:
    :param neighbors_dict: int --> (anchor_ids, anchor_counts)
    :return:
    """
    anchor_ids = neighbors_dict['anchor'][0]
    sub_graph = graph.subgraph(anchor_ids)
    return sub_graph


def add_self_loop_in_graph(graph, self_loop_r: int):
    """
    :param graph:
    :param self_loop_r:
    :return:
    """
    number_of_nodes = graph.number_of_nodes()
    self_loop_r_array = torch.full((number_of_nodes,), self_loop_r, dtype=torch.long)
    node_ids = torch.arange(number_of_nodes)
    graph.add_edges(node_ids, node_ids, {'rid': self_loop_r_array})


def sub_graph_cls_addition(subgraph, cls_parent_node_id: int, special_relation_dict: dict):
    """
    add one cls node into sub-graph as super-node
    :param subgraph:
    :param cls_parent_node_id: cls node shared across all subgraphs
    :param special_relation_dict: {cls_r: cls_r index}
    :return: sub_graph added cls_node (for graph level representation learning
    """
    assert 'cls_r' in special_relation_dict
    subgraph.add_nodes(1)  # the last node is the cls_node
    subgraph.ndata['nid'][-1] = cls_parent_node_id  # set the nid (parent node id) in sub-graph
    parent_node_ids, sub_node_ids = subgraph.ndata['nid'].tolist(), subgraph.nodes().tolist()
    parent2sub_dict = dict(zip(parent_node_ids, sub_node_ids))
    cls_idx = parent2sub_dict[cls_parent_node_id]
    assert cls_idx == subgraph.number_of_nodes() - 1
    cls_relation = [special_relation_dict['cls_r']] * (2 * (subgraph.number_of_nodes() - 1))
    cls_relation = torch.tensor(cls_relation, dtype=torch.long)
    cls_src_nodes = [cls_idx] * (subgraph.number_of_nodes() - 1)
    cls_src_nodes = torch.tensor(cls_src_nodes, dtype=torch.long)
    cls_dst_nodes = torch.arange(0, subgraph.number_of_nodes() - 1)
    cls_src, cls_dst = torch.cat((cls_src_nodes, cls_dst_nodes)), np.concatenate((cls_dst_nodes, cls_src_nodes))
    # bi-directional cls_nodes
    subgraph.add_edges(cls_src, cls_dst, {'rid': cls_relation})
    return subgraph, parent2sub_dict


def cls_sub_graph_extractor(graph, edge_dict: dict, neighbors_dict: dict, special_relation_dict: dict,
                            node_arw_label_dict: dict, bi_directed: bool = True, self_loop: bool = False, debug=False):
    """
    extract the sub-graph according to edge_dict and then add cls_node as super node
    :param graph: original large graph
    :param edge_dict: sampling edges
    :param neighbors_dict: multi-hop information
    :param special_relation_dict:
    :param node_arw_label_dict
    :param bi_directed:
    :param self_loop
    :param debug:
    :return: adding cls node and bi-directional edges with 'cls_r'
    """
    start_time = time() if debug else 0
    subgraph = sub_graph_extractor(graph=graph, edge_dict=edge_dict, bi_directed=bi_directed,
                                   neighbors_dict=neighbors_dict)
    cls_parent_node_id = neighbors_dict['cls'][0][0].data.item()
    subgraph, parent2sub_dict = sub_graph_cls_addition(subgraph=subgraph, cls_parent_node_id=cls_parent_node_id,
                                                       special_relation_dict=special_relation_dict)
    assert len(parent2sub_dict) == len(node_arw_label_dict) and len(parent2sub_dict) == subgraph.number_of_nodes()
    node_orders = torch.zeros(len(parent2sub_dict), dtype=torch.long)
    for key, value in parent2sub_dict.items():
        node_orders[value] = node_arw_label_dict[key]
    subgraph.ndata['n_rw_label'] = node_orders
    end_time = time() if debug else 0
    if self_loop:
        add_self_loop_in_graph(graph=graph, self_loop_r=special_relation_dict['loop_r'])
    if debug:
        print('CLS sub-graph construction time = {:.4f} seconds'.format(end_time - start_time))
    return subgraph, parent2sub_dict


def cls_anchor_sub_graph_augmentation(subgraph, parent2sub_dict: dict, neighbors_dict: dict,
                                      special_relation_dict: dict, edge_dir: str,
                                      bi_directed: bool = True):
    """
    :param subgraph: sub-graph with anchor-node
    :param parent2sub_dict: map parent ids to the sub-graph node ids
    :param neighbors_dict: multi-hop neighbors to anchor-node
    :param special_relation_dict: {x_hop_x_r}
    :param edge_dir: edge direction
    :param bi_directed: whether bi_directional graph
    :return: graph augmentation by randomly adding "multi-hop edges" in graphs
    """
    anchor_parent_node_id = neighbors_dict['anchor'][0][0].data.item()
    anchor_idx = parent2sub_dict[anchor_parent_node_id]
    assert anchor_idx < subgraph.number_of_nodes() - 1
    filtered_neighbors_dict = dict([(key, value) for key, value in neighbors_dict.items()
                                    if 'hop' in key and value[0].shape[0] > 0 and 'hop_1' not in key])
    if len(filtered_neighbors_dict) == 0:
        aug_sub_graph = copy.deepcopy(subgraph)
        number_of_nodes = subgraph.number_of_nodes()
        node_ids = torch.arange(number_of_nodes - 1)
        self_loop_r = torch.full((number_of_nodes - 1,), special_relation_dict['loop_r'], dtype=torch.long)
        aug_sub_graph.add_edges(node_ids, node_ids, {'rid': self_loop_r})
        assert subgraph.number_of_nodes() == aug_sub_graph.number_of_nodes()
        return aug_sub_graph
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    hop_neighbors_ = [key for key, value in filtered_neighbors_dict.items()]
    filtered_neighbor_hop_num = len(hop_neighbors_)
    view_num = random.randint(1, filtered_neighbor_hop_num + 1)
    hop_neighbor_names = random.choice(hop_neighbors_, view_num, replace=False)
    # #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    aug_sub_graph = copy.deepcopy(subgraph)
    for hop_neighbor in hop_neighbor_names:
        assert edge_dir in hop_neighbor
        relation_idx = special_relation_dict['{}_r'.format(hop_neighbor)]
        hop_neighbor_ids, hop_neighbor_freq = filtered_neighbors_dict[hop_neighbor]
        hop_neighbor_ids = torch.as_tensor([parent2sub_dict[_] for _ in hop_neighbor_ids.tolist()], dtype=torch.long)
        anchor_array = torch.full(hop_neighbor_ids.shape, anchor_idx, dtype=torch.long)
        relation_array = torch.full(hop_neighbor_ids.shape, relation_idx, dtype=torch.long)
        if edge_dir == 'in':
            src_nodes = hop_neighbor_ids
            dst_nodes = anchor_array
        else:
            src_nodes = anchor_array
            dst_nodes = hop_neighbor_ids
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if bi_directed:
            if edge_dir == 'in':
                rev_relation = '{}_r'.format(hop_neighbor.replace('in', 'out'))
            else:
                rev_relation = '{}_r'.format(hop_neighbor.replace('out', 'in'))
            rev_relation_idx = special_relation_dict[rev_relation]
            rev_relation_array = torch.full(hop_neighbor_ids.shape, rev_relation_idx, dtype=torch.long)
            aug_sub_graph.add_edges(torch.cat([src_nodes, dst_nodes]),
                                    torch.cat([dst_nodes, src_nodes]),
                                    {'rid': torch.cat([relation_array, rev_relation_array])})
        else:
            aug_sub_graph.add_edges(src_nodes, dst_nodes, {'rid': relation_array})
    return aug_sub_graph


def sub_graph_multiview_augmentation(subgraph, hop_num: int, edge_dir: str, special_entity_dict: dict,
                                     special_relation_dict: dict):
    assert edge_dir in {'in', 'out'}
    view_num = random.randint(1, hop_num + 1)
    samp_hop_nums = random.choice(np.arange(2, hop_num + 1), size=view_num, replace=False)
    hop_relations = [(_, '{}_hop_{}_r'.format(edge_dir, _)) for _ in samp_hop_nums if '{}_hop_{}_r'.format(edge_dir, _)
                     in special_relation_dict]
    assert len(hop_relations) > 0
    aug_sub_graph = copy.deepcopy(subgraph)
    for hop_num, hop_relation in hop_relations:
        hop_graph = dgl.khop_graph(g=subgraph, k=hop_num)
        src_nodes, dst_nodes = hop_graph.edges()
        relation_tid_i = torch.LongTensor(src_nodes.shape).fill_(special_relation_dict[hop_relation])

        aug_sub_graph.add_edges(src_nodes, dst_nodes, {'rid': relation_tid_i})
    cls_parent_node_id = special_entity_dict['cls']
    aug_sub_graph, _ = sub_graph_cls_addition(subgraph=aug_sub_graph, cls_parent_node_id=cls_parent_node_id,
                                              special_relation_dict=special_relation_dict)
    return aug_sub_graph
