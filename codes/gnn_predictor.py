from codes.gnn_encoder import GraphSimSiamEncoder
from torch import nn
import torch

class NodeClassificationModel(nn.Module):
    def __init__(self, graph_encoder: GraphSimSiamEncoder, encoder_dim: int, num_of_classes: int,
                 fix_encoder: bool = True):
        super(NodeClassificationModel, self).__init__()
        self.graph_encoder = graph_encoder
        self.predictor = nn.Linear(in_features=encoder_dim, out_features=num_of_classes)
        self.fix_encoder = fix_encoder

    @staticmethod
    def get_representation(sub_model: GraphSimSiamEncoder, batch: dict, fix_encoder: bool=True):
        if fix_encoder:
            with torch.no_grad():
                graph_emb = sub_model.encode(batch=batch)
        else:
            graph_emb = sub_model.encode(batch=batch)
        return graph_emb

    def forward(self, batch):
        graph_embed = self.get_representation(sub_model=self.graph_encoder, batch=batch, fix_encoder=self.fix_encoder)
        pred_scores = self.predictor(graph_embed)
        return pred_scores
