import torch
import torch.nn as nn
from torch import Tensor


class MyAtten(nn.Module):
    def __init__(self,query_size,key_size,c_size,output_size):
        """
        :param query_size: 张量Q最后一个维度的形状，目前这里就是8
        :param key_size: 张量K最后一个维度的形状，目前这里就是8
        :param c_size: 张量C最后一个维度的形状，目前这里就是8
        :param output_size: Q和C拼接后，并且经过线性调整形状后的张量中最后一个维度的形状，目前这里就是8
        """

        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.query_size = query_size
        self.key_size = key_size
        self.c_size = c_size
        self.output_size = output_size

        # 3- 搭建网络结构
        # 3.1- 计算QK相似性的线性层。对应原理图中的第2步
        self.score_linear = nn.Linear(in_features=self.query_size+self.key_size,out_features=1)
        # self.score_linear = nn.Linear(in_features=16,out_features=1)

        # 3.2- 调整Q和C拼接后的张量形状线性层。对应原理图中的第6步
        self.qc_linear = nn.Linear(in_features=self.query_size+self.c_size,out_features=self.output_size)
        # self.qc_linear = nn.Linear(in_features=16,out_features=8)

    def forward(self,Q:Tensor,K,V):
        """
        目前暂时不研究QKV如何来的。只计算数据准备好后的注意力的计算过程。对应原理图中中间的绿色框
        :param Q: query，张量形状[batch_size,num_layers,hidden_size] -> [2,1,8]
        :param K: key，张量形状[batch_size,seq_len,hidden_size] -> [2,5,8]
        :param V: value，与key完全一致，张量形状[batch_size,seq_len,hidden_size] -> [2,5,8]
        :return:
        """
        # 1- Q和K拼接
        # 1.1- Q和K目前不能直接拼接，需要将Q的形状 由 [2,1,8] -> [2,5,8]
        seq_len = K.shape[1]
        # expand：扩大张量的形状。-1表示调整前后对应位置的形状不变
        Q_expand = Q.expand(size=(-1,seq_len,-1))
        print(f"1-->{Q.shape}") # [2,1,8]
        print(f"2-->{Q_expand.shape}") # [2,5,8]

        # 1.2- 拼接：Q_expand [2,5,8] K [2,5,8] -> [2,5,16]
        qk_cat = torch.cat(tensors=(Q_expand,K), dim=-1)
        print(f"3-->{qk_cat.shape}") # [2,5,16]

        # 2- 使用线性层对QK拼接后的结果计算相似性
        score = self.score_linear(qk_cat)
        print(f"4-->{score.shape}") # [2,5,1]

        # 3- 将相似性转成权重矩阵
        # 3.1- 调整score的形状：由[2,5,1] -> [2,1,5]
        score = score.transpose(dim0=1,dim1=2)
        print(f"5-->{score.shape}") # [2,1,5]

        # 3.2- 调用Softmax
        weighted_mat = torch.softmax(score, dim=-1)
        print(f"6-->{weighted_mat.shape}") # [2,1,5]

        # 4- 加权求和：权重矩阵和V进行矩阵乘法。计算得到专属信息包
        # weighted_mat [2,1,5]；V [2,5,8] -> [2,1,8]
        C = torch.bmm(weighted_mat, V)
        print(f"7-->{C.shape}") # [2,1,8]

        # 5- Q和C拼接：Q [2,1,8]；C [2,1,8] -> [2,1,16]
        qc_cat = torch.cat(tensors=(Q,C), dim=-1)
        print(f"8-->{qc_cat.shape}") # [2,1,16]

        # 6- 对拼接后的张量进行形状调整，以满足Decoder端GRU输入数据的要求
        result = self.qc_linear(qc_cat)
        print(f"9-->{result.shape}") # [2,1,8]

        return result,weighted_mat

if __name__ == '__main__':
    # 因为只讲PyTorch中注意力是如何计算得到。因此拼接后的value、Q、V我们直接手动初始化

    # 1- 定义基础变量
    batch_size = 2      # 每个批次中句子的条数
    seq_len = 5         # 每个句子中词的个数
    hidden_size = 8     # 我们人为设置词向量维度和隐藏向量维度相同
    num_layers = 1      # 隐藏层层数为1

    # 2- 初始化Q、K、V
    """
        为什么Q的形状中间是1？
        答：当前只针对Decoder解码器端一个时间步。当前时刻拿一个Q，去和Encoder全部位置算相似性
    """
    Q = torch.randn(size=(batch_size,num_layers,hidden_size))   # Decoder端当前时间步的隐藏状态
    K = torch.randn(size=(batch_size,seq_len,hidden_size))      # Encoder端多个词的隐藏状态拼接后的
    V = torch.randn(size=(batch_size,seq_len,hidden_size))      # Encoder端多个词的隐藏状态拼接后的

    # 3- 计算注意力
    query_size = hidden_size
    key_size = hidden_size
    c_size = hidden_size
    output_size = hidden_size

    attn_model = MyAtten(query_size,key_size,c_size,output_size)
    output,attn_weight = attn_model(Q,K,V)

    print(f"权重值{attn_weight}")
    print(f"权重值和{attn_weight.sum()}")

