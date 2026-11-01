import torch
import torch.nn as nn

def rnn_demo(num_layers):
    # 1- 创建RNN循环网络层
    """
        参数解释：
            input_size：输入数据的向量维度
            hidden_size：隐藏状态向量维度
            num_layers：隐藏层的层数
    """
    rnn = nn.RNN(input_size=4,hidden_size=5,num_layers=num_layers)

    # 2- 准备数据
    # 2.1- 本次输入数据
    input = torch.randn(size=(3,2,4))
    # 2.2- 上一个时间步的隐藏状态。
    h0 = torch.randn(size=(num_layers,2,5))

    # 3- 调用RNN
    """
    #
        
        输入参数的张量形状：
            本次输入数据input：[seq_len每条句子中词的个数, batch_size每个批次中样本的条数, input_size词向量维度]
            上一个时间步的隐藏状态h0：[num_layers隐藏层的层数, batch_size每个批次中样本的条数, hidden_size隐藏状态向量维度]
        
        返回值的张量形状：
            本次时间步的预测结果output：[seq_len每条句子中词的个数, batch_size每个批次中样本的条数, hidden_size隐藏状态向量维度]
            本次更新后的隐藏状态hidden：[num_layers隐藏层的层数, batch_size每个批次中样本的条数, hidden_size隐藏状态向量维度]
        这里全是三维的张量！！！
    """
    output,hidden = rnn(input,h0)  

    # 4- 打印结果
    print(f"output的形状-->{output.shape}")    # [3,2,5]
    print(f"hidden的形状-->{hidden.shape}")    # [num_layers,2,5]

    print(f"output的内容-->\n{output}")
    print(f"hidden的内容-->\n{hidden}")


if __name__ == '__main__':
    # rnn_demo(num_layers=1)
    rnn_demo(num_layers=2)