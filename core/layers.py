import torch
from torch import Tensor, LongTensor
from torch import nn
from dgl.nn.pytorch.utils import Identity


class EmbeddingLayer(nn.Module):
    def __init__(self, num: int, dim: int, project_dim: int = None):
        super(EmbeddingLayer, self).__init__()
        self.num = num
        self.dim = dim
        self.proj_dim = project_dim
        self.embedding = nn.Embedding(num_embeddings=num, embedding_dim=dim)
        if self.proj_dim is not None and self.proj_dim > 0:
            self.projection = torch.nn.Linear(self.dim, self.proj_dim, bias=False)
        else:
            self.projection = Identity()

    def init_with_tensor(self, data: Tensor, freeze=False):
        self.embedding = nn.Embedding.from_pretrained(embeddings=data, freeze=freeze)

    def init(self, emb_init=0.1):
        """Initializing the embeddings.
        Parameters
        ----------
        emb_init : float
            The initial embedding range should be [-emb_init, emb_init].
        """
        nn.init.xavier_normal_(self.embedding.weight, emb_init)
        gain = nn.init.calculate_gain('relu')
        if isinstance(self.projection, nn.Linear):
            nn.init.xavier_normal_(self.projection.weight, gain=gain)

    def _embed(self, embeddings):
        embeddings = self.projection(embeddings)
        return embeddings

    def forward(self, indexes: LongTensor):
        embed_data = self._embed(self.embedding(indexes))
        return embed_data
