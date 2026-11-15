import torch
import math
from torch import Tensor

from input import *
import copy

def attention(query:Tensor,key:Tensor,value:Tensor,mask=None,dropout=None):
    """
    实现三种注意力机制
    1.编码器 多头自注意力机制
    2.解码器 多头自注意力机制 mask不为空
    3.解码器 多头交叉注意力机制 query来着编码器 key，value来着解码器
    """

    # 1.获取词向量的维度 world_dim
    world_dim=query.shape[-1]
    
    # 2.Q和K的转置（3维转置：交换最后两个维度）相乘，除以根号下的词语维度 
    score=torch.matmul(query,key.transpose(dim0=-1,dim1=-2))/math.sqrt(world_dim)

    # 3.使用掩码mask
    if mask is not None:
        score.masked_fill(mask==0,value=-1e9)

    # 4.将相似性分数softMax转化为权值
    weight=torch.softmax(score,dim=-1)

    # 5.随机失活
    if dropout is not None:
        weight=dropout(weight)

    # 6.权重矩阵和N进行乘法运算，得到专属信息包
    C=torch.matmul(weight,value)

    return C,weight

def use_attention():
    # 1- 输入的数据先经过 词嵌入层 和 位置编码
    posi_embed = use_positional_encoding()

    # 2- 调用注意力计算方法
    # 2.1- 准备query、key、value参数
    query=key=value=posi_embed

    # 2.2- 准备掩码【可选】
    # (2,4,4)值的来源于use_positional_encoding的x。每个批次2条句子，每条句子4个词
    # 形状：[batch_size,seq_len,seq_len]
    mask = torch.triu(torch.ones(size=(2,4,4)))

    # 2.3- 随机失活网络层
    dropout = nn.Dropout(p=0.1)

    # 2.4- 调用
    # 2.4.1- 编码器端：没有掩码的多头自注意力
    encoder_attention_C, encoder_attention_weight = attention(query,key,value,dropout=dropout)
    print(f"编码器C：{encoder_attention_C.shape}-->{encoder_attention_C}")
    print(f"编码器C权重：{encoder_attention_weight.shape}-->{encoder_attention_weight}")
    print('='*30)

    # 2.4.2- 解码器端：有掩码的多头自注意力
    decoder_attention_C, decoder_attention_weight = attention(query,key,value,dropout=dropout,mask=mask)
    print(f"解码器C：{decoder_attention_C.shape}-->{decoder_attention_C}")
    print(f"解码器C权重：{decoder_attention_weight.shape}-->{decoder_attention_weight}")

def clone(module_obj,number):
    return  nn.ModuleList([copy.deepcopy(module_obj) for index in range(number)])

# 多头注意力实现
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,head,dropout_p=0.2):

        self.d_model=d_model
        self.head=head
        self.dropout=nn.Dropout(dropout_p)
        self.head_dim=self.d_model//self.head

        self.liner_list=clone(module_obj=nn.Linear(in_features=self.d_model,out_features=self.d_model),number=4)

    def forward(self,query,key,value,mask:Tensor):
        if mask is not None: 
            mask=mask.unsqueeze(dim=0)
            
        batch_size=query.shape[0]

        """
            #zip的作用是将把多个序列（列表、元组等）里的元素，按位置序号“一对一”咬合配对，打包成一个个小元组。
            names = ["张三", "李四", "王五"]
            scores = [90, 85, 95]
            result = list(zip(names, scores))
            # 结果：[('张三', 90), ('李四', 85), ('王五', 95)]
        """ 
        model_and_data_pair=list(zip(self.liner_list,(query,key,value)))
        handle_result_list=[]

        for model,data in model_and_data_pair:
            model_output:Tensor=model(data)

            tem_t= model_output.reshape(batch_size,-1,self.head,self.head_dim).transpose(dim0=1,dim1=2)
            handle_result_list.append(tem_t)
            

    new_query,new_key,new_value=handle_result_list

        #多个头并行计算注意力机制
    C,weight= attention(new_key,new_query,new_value, mask ,self.dropout)

        # 张量降维
    result=C.transpose(dim0=1,dim1=2).reshape(batch_size,-1,self.head*self.head_dim)

        # 最后一个线性层对多头结果进行处理
    return self.liner_list[-1](result)

# 测试多头注意力计算
def use_multi_head_attention():
    # 1- 获取位置编码之后的词数据
    position_data = use_positional_encoding()

    # 2- query、key、value参数
    query=key=value=position_data

    # 3- 创建多头注意力实例对象
    mask = torch.triu(torch.ones(size=(8, 4, 4)))
    my_attention = MultiHeadAttention(d_model=512,head=8,dropout_p=0.1)

    # 4- 调用前向传播
    result = my_attention(query,key,value,mask=mask)
    print(f"多头注意力计算结果：{result.shape}")
    print(f"多头注意力计算结果：{result}")

    return result



if __name__=='__main__':
    use_attention()
