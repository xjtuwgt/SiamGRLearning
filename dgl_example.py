import dgl
import torch

def add_self_loop_in_graph(graph, self_loop_r: int):
    """
    :param graph:
    :param self_loop_r:
    :return:
    """
    number_of_nodes = graph.number_of_nodes()
    self_loop_r_array = torch.full((number_of_nodes,), self_loop_r, dtype=torch.long)
    node_ids = torch.arange(number_of_nodes)
    graph.add_edges(node_ids, node_ids, {'e_x': self_loop_r_array})

g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
g.edata['e_x'] = torch.as_tensor([100, 200])
x = torch.as_tensor([0, 0])
y = torch.as_tensor([2,2])
z = torch.as_tensor([5000, 5000])
g.add_edges(x, y, {'e_x': z})
print(g)

add_self_loop_in_graph(graph=g, self_loop_r=10000)
print(g)

