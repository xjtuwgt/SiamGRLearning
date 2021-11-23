import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, base_encoder_out_dim: int, dim=1024, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the projector (default: 512)
        """
        super(SimSiam, self).__init__()
        # create the encoder = base_encoder + a 3-layer projector
        self.prev_dim = base_encoder_out_dim
        self.graph_encoder = base_encoder
        self.mapper = nn.Sequential(nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                                     nn.BatchNorm1d(self.prev_dim),
                                     nn.ReLU(inplace=True),  # first layer
                                     nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                                     nn.BatchNorm1d(self.prev_dim),
                                     nn.ReLU(inplace=True),  # second layer
                                     nn.Linear(self.prev_dim, dim, bias=False),
                                     nn.BatchNorm1d(dim, affine=False))  # output layer
        # build a 2-layer projection
        self.projector = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x1, x2, cls_or_anchor='cls'):
        """
        Input:
            x1: first views of input
            x2: second views of input
            cls_or_anchor: 'cls' or 'anchor'
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        # compute features for one view
        g1 = self.graph_encoder(x1, cls_or_anchor)
        g2 = self.graph_encoder(x2, cls_or_anchor)
        z1 = self.mapper(g1)  # NxC
        z2 = self.mapper(g2)  # NxC

        p1 = self.projector(z1)  # NxC
        p2 = self.projector(z2)  # NxC
        return p1, p2, z1.detach(), z2.detach()

    def encode(self, x, cls_or_anchor='cls'):
        g = self.graph_encoder(x, cls_or_anchor)
        z = self.mapper(g)
        return z
