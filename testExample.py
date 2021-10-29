from core.layers import EmbeddingLayer
import torch

embed_layer = EmbeddingLayer(num=10, dim=3)
idxes = torch.LongTensor([1,2,3])

print(embed_layer)
print(embed_layer(idxes).shape)