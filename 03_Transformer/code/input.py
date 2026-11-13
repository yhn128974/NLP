import torch.nn as nn
import math
class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model):

        super().__init__

        self.vocab_size=vocab_size
        self.d_model=d_model

        self.edb=nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)

    def forward(self,input):
        # 乘以math.sqrt(self.d_model)是为了防止词向量的值与位置编码的值相差太大（当词向量我维度很大时可能会出现很小的值）
        return self.edb(input)* math.sqrt(self.d_model)
