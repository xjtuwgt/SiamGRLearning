from core.gnn_layers import GDTLayer, RGDTLayer
from core.siamese_builder import SimSiam
from core.layers import EmbeddingLayer
from torch import nn

class GDTEncoder(nn.Module):
    def __init__(self, config):
        super(GDTEncoder, self).__init__()
        self.config = config
        self.node_embed_layer = EmbeddingLayer(num=self.config.node_number, dim=self.config.node_emb_dim)
        self.relation_embed_layer = EmbeddingLayer(num=self.config.relation_number, dim=self.config.relation_emb_dim)
        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(module=RGDTLayer(in_ent_feats=self.config.node_emb_dim,
                                                   in_rel_feats=self.config.relation_emb_dim,
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
    def forward(self, batch_g_pair):
        batch_g, batch_cls = batch_g_pair
        ent_ids = batch_g.ndata['nid']
        rel_ids = batch_g.edata['rid']
        ent_features = self.node_embed_layer(ent_ids)
        rel_features = self.relation_embed_layer(rel_ids)
        with batch_g.local_scope():
            h = ent_features
            for l in range(self.num_layers):
                if l == 0:
                    h = self.gdt_layers[l](batch_g, h, rel_features)
                else:
                    h = self.gdt_layers[l](batch_g, h)
            graph_cls_embed = h[batch_cls]
            return graph_cls_embed

class GraphSimSiamEncoder(nn.Module):
    def __init__(self, config):
        super(GraphSimSiamEncoder, self).__init__()
        self.config = config
        self.graph_encoder = GDTEncoder(config=config)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.graph_siam_encoder = SimSiam(base_encoder=self.graph_encoder,
                                          base_encoder_out_dim=self.config.hidden_dim,
                                          dim=self.config.siam_dim,
                                          pred_dim=self.config.siam_pred_dim)
    def forward(self, batch):
        p1, p2, z1, z2 = self.graph_siam_encoder(batch['batch_graph_1'], batch['batch_graph_2'])
        return p1, p2, z1, z2