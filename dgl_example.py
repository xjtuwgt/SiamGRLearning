import dgl
from codes.citation_graph_data import citation_graph_reconstruction
import torch
# from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
import matplotlib.pyplot as plt
temp_file_name = '/Users/wangguangtao/Desktop/desktop/temp.txt'
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

# g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
# g.edata['e_x'] = torch.as_tensor([100, 200])
# x = torch.as_tensor([0, 0])
# y = torch.as_tensor([2,2])
# z = torch.as_tensor([5000, 5000])
# g.add_edges(x, y, {'e_x': z})
# print(g)
#
# add_self_loop_in_graph(graph=g, self_loop_r=10000)
# print(g)
train_loss_list = []
with open(temp_file_name) as file:
    lines = file.readlines()
    for line in lines:
        if 'Train_loss' in line:
            tokens = line.split(':')
            metric = float(tokens[-1].strip())
            train_loss_list.append(metric)
            # print(line)
            # print(metric)
x = (np.arange(len(train_loss_list)) + 1) * 5
train_loss = np.array(train_loss_list)
# for _ in train_loss:
#     print(_)
# print(train_loss)
plt.plot(x, train_loss)
plt.show()

# # citation_graph_reconstruction(dataset='pubmed')
# from dgl.data.citation_graph import PubmedGraphDataset, CiteseerGraphDataset
#
# data = CiteseerGraphDataset()