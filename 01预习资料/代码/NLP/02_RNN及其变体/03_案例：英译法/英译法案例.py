import random
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1- 设置变量
# 运行设备
device = ("cuda" if torch.cuda.is_available() else "cpu")   # Windows的写法
# device = "mps"  # Mac的写法

# 数据文件路径
data_path = "../data/eng-fra-v2.txt"

# 句子开始标识
SOS_TOKEN = 0

# 句子结束标识
EOS_TOKEN = 1

# 句子长度规范：长度的上限
MAX_LENGTH = 10

# 2- 数据清洗
def normalizeString(line):
    # 将文本全部转小写
    line = line.lower().strip()
    # print(f"-{line}-")

    # 在指定的标点符号的前面增加空格
    line = re.sub(pattern=r"([.!?])",repl=r" \1",string=line)

    # 去除除了26个小写、.!?以外的其他符号，全部替换成空格
    line = re.sub(pattern=r"[^a-z.!?]+", repl=r" ", string=line)

    return line

# 3- 构建词汇表
def getdata():
    # 1- 读取文件
    with open(data_path,encoding="UTF-8",mode="r") as f:
        eng_fra_list = f.readlines()

    # 2- 划分得到英语和法语
    # eng_fra_pairs结构：[[英语句子1, 法语句子1], [英语句子2, 法语句子2]...]
    eng_fra_pairs = []
    for eng_fra_line in eng_fra_list:

        tmp_list = []
        for line in eng_fra_line.split("\t"):
            tmp_list.append(normalizeString(line))

        eng_fra_pairs.append(tmp_list)

    # 3- 分别得到英语词汇表和法语词汇表
    english_word2index = {"SOS":SOS_TOKEN,"EOS":EOS_TOKEN}
    english_word_cnt = 2

    franch_word2index = {"SOS":SOS_TOKEN,"EOS":EOS_TOKEN}
    franch_word_cnt = 2

    for eng_fra_list in eng_fra_pairs:
        # 英语句子处理：分词->去重->添加
        for eng_word in eng_fra_list[0].split(" "):
            if eng_word not in english_word2index:
                english_word2index[eng_word] = english_word_cnt
                english_word_cnt += 1

        # 法语句子处理：分词->去重->添加
        for fra_word in eng_fra_list[1].split(" "):
            if fra_word not in franch_word2index:
                franch_word2index[fra_word] = franch_word_cnt
                franch_word_cnt += 1

    # 4- 将法语字典中key和value的位置交换
    franch_index2word = {index:word for word,index in franch_word2index.items()}

    return english_word2index,english_word_cnt,franch_word2index,franch_word_cnt,eng_fra_pairs,franch_index2word

english_word2index,english_word_cnt,franch_word2index,franch_word_cnt,eng_fra_pairs,franch_index2word = getdata()

# 4- 自定义Dataset
class MyPairsDataset(Dataset):
    def __init__(self,eng_fra_pairs):
        self.eng_fra_pairs = eng_fra_pairs
        self.sample_cnt = len(eng_fra_pairs)    # 样本条数

    def __len__(self):
        return self.sample_cnt

    def __getitem__(self, index):
        # 1- 防止索引为负数；防止索引越界
        index = min(max(index,0), self.sample_cnt - 1)

        # 2- 获得英语句子（特征数据）和法语句子（目标值）
        # 2.1- 获得句子。[英语句子1, 法语句子1]
        eng_fra_pair = self.eng_fra_pairs[index]

        # 句子 -> 分词 -> 变成索引
        """
            为什么只在句子的末尾增加结束标识EOS_TOKEN，而不增加SOS_TOKEN？
            答：
                1- Seq2Seq架构要求必须要有EOS_TOKEN，而SOS_TOKEN可选的
                2- SOS_TOKEN会在Decoder中进行数据处理的时候再增加
                3- Decoder端我们只能通过添加SOS_TOKEN告诉解码器开始进行翻译的工作；以及通过添加EOS_TOKEN告诉解码器翻译工作结束
        """
        # 2.2- 处理英语句子
        x = [english_word2index[word] for word in eng_fra_pair[0].split(" ")]
        # 所有的张量数据必须放在同一个硬件设备上，要么同时在GPU，要么同时在CPU
        x_tensor = torch.tensor(x,dtype=torch.long,device=device)
        # print(f"原始英语句子{eng_fra_pair[0]}，词索引向量{x_tensor}")

        # 2.3- 处理法语句子
        y = [franch_word2index[word] for word in eng_fra_pair[1].split(" ")]
        y.append(EOS_TOKEN)
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)
        # print(f"原始法语句子{eng_fra_pair[1]}，词索引向量{y_tensor}")

        return x_tensor,y_tensor

# 5- 创建DataLoader
def get_dataloader():
    # 1- 创建自定义Dataset实例对象
    dataset = MyPairsDataset(eng_fra_pairs)

    # 2- 创建数据加载器
    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=True)

    return dataloader

# 6- 无注意力机制的编码器
class Encoder(nn.Module):
    def __init__(self,eng_vocab_size,input_size,hidden_size):
        """
        :param eng_vocab_size: 英语词汇表大小
        :param input_size: 词向量维度
        :param hidden_size: 隐藏状态向量维度
        """

        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.eng_vocab_size = eng_vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 3- 搭建网络结构
        # 3.1- 词嵌入层
        self.ebd = nn.Embedding(num_embeddings=self.eng_vocab_size,embedding_dim=self.input_size)

        # 3.2- 循环网络层
        self.gru = nn.GRU(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=1,batch_first=True)

    def forward(self,input,hidden):
        """
        Encoder端的前向传播：负责理解英语句子的语义信息
        :param input: 本次输入数据，张量形状：[batch_size每个批次中英语句子的条数,seq_len英语句子中词的个数]
        :param hidden: 上一个时间步的隐藏状态，张量形状：[num_layers,batch_size,hidden_size]
        :return:
        """
        # 1- 调用词嵌入层
        # embed形状：[batch_size每个批次中英语句子的条数,seq_len英语句子中词的个数,input_size词向量维度]
        # print(f"1--input-->{input.shape}")
        embed = self.ebd(input)
        # print(f"2--embed-->{embed.shape}")

        # 2- 调用GRU
        output,hidden = self.gru(embed,hidden)

        return output,hidden

    def init_hidden(self):
        # 隐藏状态张量形状：[num_layers,batch_size,hidden_size]
        # num_layers：目前手动指定为1层
        # batch_size：因为我们没有对英语句子进行长度规范的处理，因此只能设置为1
        return torch.zeros(size=(1,1,self.hidden_size)).to(device=device)

# 测试编码器
def use_encoder():
    # 1- 准备数据
    dataloader = get_dataloader()

    # 2- 创建编码器对象
    my_encoder = Encoder(eng_vocab_size=english_word_cnt,input_size=256,hidden_size=256)
    my_encoder.to(device=device)

    # 3- 遍历数据，前向传播
    for x,y in dataloader:
        # 3.1- 初始隐藏状态
        hidden = my_encoder.init_hidden()

        # 3.2- 前向传播
        output,hidden = my_encoder(x,hidden)

        print(f"output-->{output.shape}")
        print(f"hidden-->{hidden.shape}")

        break

# 7- 解码器：无注意力机制
class Decoder(nn.Module):
    def __init__(self,fra_vocab_size,hidden_size):
        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.fra_vocab_size = fra_vocab_size
        self.hidden_size = hidden_size

        # 3- 搭建网络结构
        # 3.1- 词嵌入层
        self.ebd = nn.Embedding(num_embeddings=self.fra_vocab_size,embedding_dim=self.hidden_size)

        # 3.2- GRU层
        self.gru = nn.GRU(input_size=self.hidden_size,hidden_size=self.hidden_size,num_layers=1,batch_first=True)

        # 3.3- 输出层
        self.output_linear = nn.Linear(in_features=self.hidden_size,out_features=self.fra_vocab_size)

        # 4- 输出层的激活函数
        # CrossEntropyLoss = LogSoftmax + NLLLoss
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self,input,hidden):
        """
        Decoder前向传播：负责翻译得到法语词
        :param input: 本次输入数据，每次只送一个法语词进来，张量形状：[batch_size,seq_len]，[[一个法语词的词索引]]
        :param hidden: 上一个时间步的隐藏状态，张量形状：[num_layers,batch_size,hidden_size]
        :return:
        """

        # 1- 调用词嵌入层
        embed = torch.relu(self.ebd(input))

        # 2- 调用GRU
        output,hidden = self.gru(embed,hidden)
        print(f"1-->output-->{output.shape}")

        # 3- 调用输出层
        # 3.1- 将output由3维变成2维，[batch_size,seq_len,hidden_size] -> [batch_size,hidden_size]
        """
            output[:,-1,:] 针对文本分类的业务场景。-1表示取最后一个时间步的隐藏状态，文本分类需要对一整条句子的完整信息理解完以后再做分类
            output[:,0,:]  针对文本翻译的业务场景。
        """
        output = output[:,0,:]
        print(f"2-->output[:,0,:]-->{output.shape}")

        # 3.2- 调用
        output = self.log_softmax(self.output_linear(output))
        print(f"3-->output-->{output.shape}")

        return output,hidden

    def init_hidden(self):
        return torch.zeros(size=(1,1,self.hidden_size)).to(device=device)

# 测试解码器
def use_decoder():
    # 1- 准备数据
    dataloader = get_dataloader()

    # 2- 创建模型实例对象
    my_encoder = Encoder(eng_vocab_size=english_word_cnt,input_size=256,hidden_size=256).to(device=device)
    my_decoder = Decoder(fra_vocab_size=franch_word_cnt,hidden_size=256).to(device=device)

    # 3- 前向传播
    for x,y in dataloader:
        # x：一条英语句子，[[词索引1, 词索引2,... ]]
        # y：一条法语句子，[[词索引1, 词索引2,... ]]

        # 3.1- 调用编码器：让模型理解英语句子的完整含义
        h0 = my_encoder.init_hidden()
        encoder_output,hidden = my_encoder(x,h0)

        # 3.2- 调用解码器：每次送入一个词
        for index in range(y.shape[1]):
            # 3.3- 解码器前向传播
            fra_word = y[0][index].reshape(1,-1)
            decoder_output,hidden= my_decoder(fra_word, hidden)

            print(f"法语词词索引{fra_word}，隐藏状态信息{hidden.shape}")

        break

# 8- 解码器：有注意力机制
class AttenDecoder(nn.Module):
    def __init__(self,fra_vocab_size,hidden_size):
        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.fra_vocab_size = fra_vocab_size
        self.hidden_size = hidden_size      # 有两个作用：1- 词向量维度；2- 隐藏状态向量维度

        # 3- 搭建网络结构
        # 3.1- 词嵌入层
        self.embedding = nn.Embedding(num_embeddings=self.fra_vocab_size,embedding_dim=self.hidden_size)

        # 3.2- Dropout随机失活层
        self.dropout = nn.Dropout(p=0.2)

        # 3.3- 线性层：用来计算QK的相似性
        self.attn = nn.Linear(in_features=self.hidden_size+self.hidden_size,out_features=MAX_LENGTH)

        # 3.4- 线性层：对Q和C拼接后的张量形状进行调整
        self.attn_combine = nn.Linear(in_features=self.hidden_size+self.hidden_size,out_features=self.hidden_size)

        # 3.5- GRU循环网络层
        self.gru = nn.GRU(input_size=self.hidden_size,hidden_size=self.hidden_size,num_layers=1,batch_first=True)

        # 3.6- 输出层
        self.out = nn.Linear(in_features=self.hidden_size,out_features=self.fra_vocab_size)

        # 激活函数
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, prev_hidden, encoder_outputs):
        """
        :param input: 解码器本次输入数据，也就是Q。张量形状 [batch_size,seq_len]，每次只传一个词进来
        :param prev_hidden: 解码器上一个时间步的隐藏状态，也就是K。张量形状 [num_layers,batch_size,hidden_size]
        :param encoder_outputs: 编码器端所有时间步隐藏状态的拼接，也就是V，而且还做了句子长度规范的处理。张量形状 [batch_size,MAX_LENGTH,hidden_size]
        :return:
        """

        # 1- 调用词嵌入层
        embedded = self.dropout(self.embedding(input))

        # 2- 计算注意力机制
        # 2.1- Q和K拼接
        # embedded就是Q
        # 如果不进行张量形状调整，那么就是num_layers与batch_size进行拼接，业务层面是不合理。
        # 调整K的形状 由 [num_layers,batch_size,hidden_size] -> [batch_size,num_layers,hidden_size]
        K = prev_hidden.transpose(dim0=0,dim1=1)
        # 拼接
        qk_cat = torch.cat(tensors=(embedded,K), dim=-1)

        # 2.2- 通过线性层算QK的相似性
        score = self.attn(qk_cat)

        # 2.3- 相似性转成权重
        attn_weights = torch.softmax(score, dim=-1)

        # 2.4- 得到专属信息包C：权重矩阵和V进行矩阵乘法运算
        # C对应原理图中的attn_applied
        C = torch.bmm(attn_weights, encoder_outputs)

        # 2.5- Q和C拼接
        qc_cat = torch.cat(tensors=(embedded,C),dim=-1)

        # 2.6- 通过线性层调整QC拼接后的张量
        qc_output = torch.relu(self.attn_combine(qc_cat))

        # 3- 调用GRU
        output,hidden = self.gru(qc_output,prev_hidden)

        # 4- 调用输出层
        output = output[:,0,:]
        output = self.log_softmax(self.out(output))

        return output,hidden,attn_weights

    def init_hidden(self):
        return torch.zeros(size=(1,1,self.hidden_size)).to(device=device)

# 9- 模型训练
# 9.1- 单次训练过程
def train_iter(x,y,my_encoder,my_decoder,encoder_adam,decoder_adam,loss):
    """
    单次样本数据的训练过程
    :param x: 英语句子，张量形状[batch_size,seq_len]
    :param y: 法语句子，张量形状[batch_size,seq_len]
    :param my_encoder: 编码器对象
    :param my_decoder: 解码器对象
    :param encoder_adam: 编码器的优化器
    :param decoder_adam: 解码器的优化器
    :param loss: 损失函数对象
    :return: 平均损失值
    """

    # 1- 调用编码器
    encoder_hidden = my_encoder.init_hidden()
    encoder_output,encoder_hidden = my_encoder(x,encoder_hidden)

    # 2- 调用解码器之前的准备工作
    # 2.1- 句子长度规范化处理，同时也是合并得到V
    # 初始化一个形状[1,10,256]的全0张量
    V = torch.zeros(size=(1,MAX_LENGTH,256), device=device)
    copy_seq_len = min(MAX_LENGTH, x.shape[1])
    V[:, :copy_seq_len, :] = encoder_output[:, :copy_seq_len, :]

    # 2.2- 准备变量
    # 将编码器端最后一个时间步的隐藏状态作为 初始隐藏状态 给到解码器使用
    decoder_hidden = encoder_hidden
    # 损失值
    loss_value = 0.0
    # 获得法语句子中词的个数
    y_seq_len = y.shape[1]
    # 存储法语翻译结果，而且上个时间步的预测结果会作为本次输入数据使用，这称之为自回归
    input_y = torch.tensor([[SOS_TOKEN]], device=device)
    # 教师机制的标识。True：使用教师机制，也就是告诉下个时间步【真实】的法语词内容。反之，把【预测】结果给到下个时间步
    teacher_forcing_flag = True if random.random()<0.5 else False

    # 3- 调用解码器
    if teacher_forcing_flag:
        # 使用教师机制，也就是告诉下个时间步【真实】的法语词内容
        for i in range(y_seq_len):
            # 3.1- 前向传播
            pred_output,decoder_hidden,attn_weights = my_decoder(input=input_y, prev_hidden=decoder_hidden, encoder_outputs=V)

            # 3.2- 计算损失值
            # print(f"y-->{y.shape}")
            # print(f"y[0]-->{y[0].shape}")
            # print(f"y[0][i]-->{y[0][i]}")

            y_true = y[0][i].reshape(1)

            # print(f"y_true-->{y_true.shape}")

            loss_value += loss(pred_output,y_true)

            # 3.3- 告诉下个时间步【真实】的法语词内容
            input_y = y[0][i].reshape(1,-1) # [batch_size,seq_len]

    else:
        # 普通训练机制，把【预测】结果给到下个时间步
        for i in range(y_seq_len):
            # 3.1- 前向传播
            pred_output, decoder_hidden, attn_weights = my_decoder(input=input_y, prev_hidden=decoder_hidden,
                                                                   encoder_outputs=V)

            # 3.2- 计算损失值
            y_true = y[0][i].reshape(1)
            loss_value += loss(pred_output, y_true)

            # 3.3- 获得预测结果的词索引
            pred_word_index = torch.argmax(pred_output, dim=-1)

            # 3.4- 判断是否到了句子的结尾
            if pred_word_index == EOS_TOKEN:
                break

            # 3.5- 告诉下个时间步【预测】的法语词内容
            input_y = pred_word_index.reshape(1, -1)  # [batch_size,seq_len]

    # 4- 反向传播固定代码
    encoder_adam.zero_grad()
    decoder_adam.zero_grad()

    loss_value.sum().backward()

    encoder_adam.step()
    decoder_adam.step()

    return loss_value.item() / y_seq_len

# 9.2- 整体训练过程
def train():
    # 1- 准备数据
    dataloader = get_dataloader()

    # 2- 创建类的实例对象
    # 2.1- 模型实例对象
    my_encoder = Encoder(eng_vocab_size=english_word_cnt,input_size=256,hidden_size=256).to(device=device)
    my_decoder = AttenDecoder(fra_vocab_size=franch_word_cnt,hidden_size=256).to(device=device)
    # 2.2- 优化器实例对象：与具体的实例对象有关，因此要创建单独的两个
    encoder_adam = torch.optim.Adam(params=my_encoder.parameters(),lr=1e-4)
    decoder_adam = torch.optim.Adam(params=my_decoder.parameters(),lr=1e-4)
    # 2.3- 损失函数对象：与具体的实例对象无关，内部只是通过预测和真实值计算差值而已，因此只需要创建一个
    loss = nn.NLLLoss()

    # 3- 设置模式
    my_encoder.train()
    my_decoder.train()

    # 4- 模型训练
    epochs = 1

    avg_loss_list = []  # 用来存储平均损失

    for epoch in range(epochs):
        total_loss_value = 0.0  # 总损失值

        for i,(x_train,y_train) in enumerate(tqdm(dataloader), start=1):
            # 1- 单次训练得到损失值
            loss_value = train_iter(x_train,y_train,my_encoder,my_decoder,encoder_adam,decoder_adam,loss)

            # 2- 更新统计指标
            total_loss_value += loss_value

            # 3- 每隔100个批次记录并且输出一次信息
            if i%100==0:
                # 计算平均损失值
                avg_loss = total_loss_value / i

                avg_loss_list.append(avg_loss)
                print(f"第{epoch+1}轮次，平均损失{avg_loss}")

    # 5- 保存训练好的模型
    torch.save(my_encoder.state_dict(), "model/encoder.pkl")
    torch.save(my_decoder.state_dict(), "model/decoder.pkl")

    # 6- 绘制损失值变化曲线
    plt.plot(avg_loss_list)
    plt.show()

# 10- 模型预测
# 10.1- 单条句子的翻译
def seq2seq_evaluate(x_tensor,my_encoder,my_decoder):
    """
    单条句子的翻译
    :param x_tensor: 一条处理后的英语句子，形状[batch_size,seq_len]
    :param my_encoder: 编码器
    :param my_decoder: 解码器
    :return: 翻译的法语词列表
    """

    with torch.no_grad():
        # 1- 调用编码器
        encoder_output,encoder_hidden = my_encoder(x_tensor,my_encoder.init_hidden())

        # 2- 调用解码器之前的准备工作
        # 2.1- 句子长度规范化处理，同时也是合并得到V
        # 初始化一个形状[1,10,256]的全0张量
        V = torch.zeros(size=(1,MAX_LENGTH,256), device=device)
        copy_seq_len = min(MAX_LENGTH, x_tensor.shape[1])
        V[:, :copy_seq_len, :] = encoder_output[:, :copy_seq_len, :]

        # 2.2- 准备变量
        # 将编码器端最后一个时间步的隐藏状态作为 初始隐藏状态 给到解码器使用
        decoder_hidden = encoder_hidden
        # 存储单次的法语翻译结果，而且上个时间步的预测结果会作为本次输入数据使用，这称之为自回归
        input_y = torch.tensor([[SOS_TOKEN]], device=device)
        # 存储所有的翻译法语词
        franch_word_list = []

        # 3- 评估/预测
        for i in range(MAX_LENGTH):
            # 3.1- 前向传播：调用解码器
            decoder_output,decoder_hidden,attn_weights = my_decoder(input=input_y, prev_hidden=decoder_hidden, encoder_outputs=V)

            # 3.2- 获得预测概率最高的那个词的词索引
            pred_fra_index = torch.argmax(decoder_output, dim=-1).item()

            # 3.3- 判断是否遇到了句子的结束标识
            if pred_fra_index == EOS_TOKEN:
                franch_word_list.append("EOS")
                break
            else:
                # 通过索引获得法语词的内容
                fra_word = franch_index2word[pred_fra_index]
                franch_word_list.append(fra_word)

            input_y = torch.tensor([[pred_fra_index]], device=device)

    return franch_word_list

# 10.2- 完整的预测流程
def use_seq2seq_evaluate():
    # 1- 手动准备未知数据
    my_samplepairs = [
        ['i m impressed with your french .', 'je suis impressionne par votre francais .'],
        ['i m more than a friend .', 'je suis plus qu une amie .'],
        ['she is beautiful like her mother .', 'elle est belle comme sa mere .']
    ]

    # 2- 加载训练好的模型
    my_encoder = Encoder(eng_vocab_size=english_word_cnt,input_size=256,hidden_size=256).to(device=device)
    my_encoder.load_state_dict(torch.load("model/encoder.pkl"))

    my_decoder = AttenDecoder(fra_vocab_size=franch_word_cnt,hidden_size=256).to(device=device)
    my_decoder.load_state_dict(torch.load("model/decoder.pkl"))

    # 3- 设置模式
    my_encoder.eval()
    my_decoder.eval()

    # 4- 预测
    for i,pair in enumerate(my_samplepairs):
        x = pair[0] # 英语句子
        y = pair[1] # 英语句子

        # 4.1- 英语句子转成张量
        x_list = [english_word2index[word] for word in x.split(" ")]
        # 所有的张量数据必须放在同一个硬件设备上，要么同时在GPU，要么同时在CPU
        x_tensor = torch.tensor(x_list, dtype=torch.long, device=device).reshape(1,-1)

        # 4.2- 对单条句子进行评估
        franch_word_list = seq2seq_evaluate(x_tensor,my_encoder,my_decoder)

        # 4.3- 输出结果
        print(f"英语句子 {x}，真实法语句子 {y}，翻译法语句子 {' '.join(franch_word_list)}")

if __name__ == '__main__':
    # 1- 数据清洗测试
    # content = " I LOve h@@@eima! "
    # normalizeString(content)

    # 2- 读取文件
    # getdata()

    # print(english_word_cnt)
    # print(franch_word_cnt)
    # print(eng_fra_pairs[:5])
    # print(english_word2index)
    # print(franch_word2index)

    # 3- 测试Dataloader
    # dataloader = get_dataloader()
    # for x,y in dataloader:
    #     print(x)
    #     print(y)
    #
    #     break

    # 4- 测试编码器
    # use_encoder()

    # use_decoder()


    # 5- 模型训练
    train()

    # 6- 模型预测
    use_seq2seq_evaluate()










