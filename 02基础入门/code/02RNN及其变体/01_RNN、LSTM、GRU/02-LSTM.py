import torch
import torch.nn as nn

if __name__ == '__main__':
    # 1- 创建LSTM网络结构
    """
        参数解释：
            bidirectional：是否是双向的网络结构。默认是False，也就是单向的。如果设置为True，LSTM就变成Bi-LSTM
    """
    lstm = nn.LSTM(input_size=4,hidden_size=5,num_layers=1,bidirectional=False)

    # 2- 准备数据
    # input的形状：[seq_len句子中词的个数,batch_size每个批次句子的个数,input_size词向量的维度]
    input = torch.randn(size=(6,3,4))

    """
        因为最终的隐藏状态是根据细胞状态经过激活函数处理得到的，
        但是激活函数不会改变张量的形状，只会改变值
        因此：隐藏状态 和 细胞状态 的张量形状是相同的
        实际工作中，一般对隐藏状态 和 细胞状态全都使用全0初始化
    """
    h0 = torch.zeros(size=(1,3,5))
    c0 = torch.zeros(size=(1,3,5))

    # 3- 调用LSTM
    """
        output形状：[seq_len,batch_size,hidden_size]
        hidden形状：[num_layers,batch_size每个批次句子的个数,hidden_size]
    """

    # 注意：传递和接受结果，都需要将隐藏状态 和 细胞状态用元组包起来
    output,(hidden,c1) = lstm(input,(h0,c0))

    print(f"output形状-->{output.shape}") # 6,3,5
    print(f"hidden形状-->{hidden.shape}")
    print(f"c1形状-->{c1.shape}") # 1,3,5

    print(f"output-->{output}")
    print(f"hidden-->{hidden}")
    print(f"c1-->{c1}")
