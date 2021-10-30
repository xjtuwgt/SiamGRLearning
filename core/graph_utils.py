from dgl.sampling import sample_neighbors
from torch import Tensor
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
    n_relations = n_relations + 1
    special_relation_dict['mask_r'] = n_relations ### for edge mask
    number_of_nodes = graph.number_of_nodes()
    number_of_relations = n_relations
    return graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict

def add_relation_ids_to_graph(graph, edge_type_ids: Tensor):
    graph.edata['tid'] = edge_type_ids
    return graph

def sub_graph_sample(graph, anchor_node_ids: Tensor, cls_node_ids: Tensor, fanouts: list,
                     edge_dir: str = 'in', bi_direct: bool=False):
    """
    :param graph: dgl graph
    :param anchor_node_ids: LongTensor
    :param cls_node_ids: LongTensor
    :param fanouts: size = hop_number, (list, each element represents the number of sampling neighbors)
    :param edge_dir:  'in' or 'out'
    :param bi_direct: bi-directional graph or not
    :return:
    """
    assert edge_dir in {'in', 'out'}
    neighbors_dict = {'anchor': anchor_node_ids}
    neighbors_dict['cls'] = cls_node_ids ## connected to all the other nodes for graph-level representation learning
    edge_dict = {} ## sampled edge dictionary: (head, t_id, tail)
    number_of_edges = graph.number_of_edges()
    new_added_bi_direct_edge_dict, added_bi_direct_edge_idx = {}, number_of_edges
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
        if bi_direct:
            has_rev_edges = graph.has_edges_between(sg_dst, sg_src)
        else:
            has_rev_edges = [False] * len(sg_src_list)
        for eid, src_id, tid, dst_id, rev_edge_flag in zip(sg_eid_list, sg_src_list, sg_tid_list, sg_dst_list,
                                                           has_rev_edges):
            edge_dict[eid] = (src_id, tid, dst_id)
            if bi_direct:
                if rev_edge_flag:
                    rev_eid = graph.edge_ids(dst_id, src_id)
                    rev_tid = graph.edata['tid'][rev_eid].data.item()
                    edge_dict[rev_eid] = (dst_id, rev_tid, src_id)
                else:
                    if (dst_id, tid, src_id) not in new_added_bi_direct_edge_dict:
                        new_added_bi_direct_edge_dict[(dst_id, tid, src_id)] = added_bi_direct_edge_idx
                        added_bi_direct_edge_idx = added_bi_direct_edge_idx + 1
        if edge_dir == 'in':
            hop_neighbor = sg_src
        else:
            hop_neighbor = sg_dst
        neighbors_dict['{}_hop_{}'.format(edge_dir, hop)] = hop_neighbor
        hop = hop + 1
    new_added_bi_direct_edge_dict = dict([(v, k) for k, v in new_added_bi_direct_edge_dict.items()])
    return neighbors_dict, edge_dict, new_added_bi_direct_edge_dict