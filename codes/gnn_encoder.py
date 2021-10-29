from core.gnn_layers import GDTLayer
from core.siamese_builder import SimSiam
from torch import nn

class GDTEncoder(nn.Module):
    def __init__(self, config):
        super(GDTEncoder, self).__init__()
        self.config = config
        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.node_emb_dim,
                                                out_ent_feats=self.config.hidden_dim,
                                                num_heads=self.config.head_num,
                                                hop_num=self.config.hop_num,
                                                alpha=self.config.alpha,
                                                feat_drop=self.config.feat_drop,
                                                attn_drop=self.config.attn_drop,
                                                residual=self.config.residual,
                                                diff_head_tail=self.config.diff_head_tail,
                                                ppr_diff=self.config.ppr_diff))
        for _ in range(self.config.layers):
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.hidden_dim,
                                                out_ent_feats=self.config.hidden_dim,
                                                num_heads=self.config.head_num,
                                                hop_num=self.config.hop_num,
                                                alpha=self.config.alpha,
                                                feat_drop=self.config.feat_drop,
                                                attn_drop=self.config.attn_drop,
                                                residual=self.config.residual,
                                                diff_head_tail=self.config.diff_head_tail,
                                                ppr_diff=self.config.ppr_diff))
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.graph_siam_encoder = SimSiam(base_encoder=self.graph_encoder,
                                          base_encoder_out_dim=self.config.hidden_dim,
                                          dim=self.config.siam_dim,
                                          pred_dim=self.config.siam_pred_dim)

    def forward(self, batch):
        return