from dgl.data import GINDataset

# data names in GIN Dataset (``'MUTAG'``, ``'COLLAB'``, \
#  ``'IMDBBINARY'``, ``'IMDBMULTI'``, \
#  ``'NCI1'``, ``'PROTEINS'``, ``'PTC'``, \
#  ``'REDDITBINARY'``, ``'REDDITMULTI5K'``)

mutag_data = GINDataset('COLLAB', self_loop=False)
print(len(mutag_data))

for _ in range(len(mutag_data)):
    g, label = mutag_data[_]
    print(g)
    print(label)
    print(_)