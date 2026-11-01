import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import jieba
import torch.nn as nn

# 解决新版本兼容性问题：
# 1. 新版本 tensorflow 的 tf.io.gfile 缺少 get_filesystem，导致 tensorboard 报错：
#    AttributeError: module 'tensorflow._api.v2.io.gfile' has no attribute 'get_filesystem'
# 2. 若直接替换为 tb.compat.tensorflow_stub.io.gfile，其缺少 join 方法，导致：
#    AttributeError: module 'tensorboard.compat.tensorflow_stub.io.gfile' has no attribute 'join'
# 因此补充 join 方法后再注入 tf.io.gfile
import tensorflow as tf
import tensorboard.compat.tensorflow_stub.io.gfile as tb_gfile
tb_gfile.join = os.path.join
tf.io.gfile = tb_gfile


# 实验：nn.Embedding层词向量可视化分析
# 1 对句子分词 word_list
# 2 对句子word2id求my_token_list，对句子文本数值化sentence2id
# 3 创建nn.Embedding层，查看每个token的词向量数据
# 4 创建SummaryWriter对象, 可视化词向量
#   词向量矩阵embd.weight.data 和 词向量单词列表my_token_list添加到SummaryWriter对象中
#   summarywriter.add_embedding(embd.weight.data, my_token_list)
# 5 通过tensorboard观察词向量相似性
# 6 也可通过程序，从nn.Embedding层中根据idx拿词向量

def dm02_nnembeding_show():

    # 1 对句子分词 word_list
    sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    sentence2 = "我爱自然语言处理"
    sentences = [sentence1, sentence2]

    word_list = []
    for s in sentences:
        word_list.append(jieba.lcut(s))
    # print('word_list--->', word_list)

    # 2 对句子word2id求my_token_list，对句子文本数值化sentence2id
    mytokenizer = Tokenizer()
    mytokenizer.fit_on_texts(word_list)
    # print(mytokenizer.index_word, mytokenizer.word_index)

    # 打印my_token_list (转换为 list，保证兼容 SummaryWriter.add_embedding 的元数据格式)
    my_token_list = list(mytokenizer.index_word.values())
    print('my_token_list-->', my_token_list)

    # 打印文本数值化以后的句子
    sentence2id = mytokenizer.texts_to_sequences(word_list)
    print('sentence2id--->', sentence2id, len(sentence2id))

    # 3 创建nn.Embedding层
    embd = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)
    # print("embd--->", embd)
    # print('nn.Embedding层词向量矩阵-->', embd.weight.data, embd.weight.data.shape, type(embd.weight.data))

    # 4 创建SummaryWriter对象 (重要：Windows环境下路径含中文会导致TensorBoard后端扫描报错，故指定纯英文日志路径)
    log_dir = r"C:\Users\YuEth\tensorboard_runs"
    summarywriter = SummaryWriter(log_dir=log_dir)
    summarywriter.add_embedding(embd.weight.data, my_token_list)
    summarywriter.close()

    # 5 通过tensorboard观察词向量相似性
    # 终端执行纯英文路径命令启动：
    # tensorboard --logdir=C:\Users\YuEth\tensorboard_runs --port=6006
    # 浏览器打开：http://127.0.0.1:6006

    print('从nn.Embedding层中根据idx拿词向量')
    # # 6 从nn.Embedding层中根据idx拿词向量
    for idx in range(len(mytokenizer.index_word)):
        tmpvec = embd(torch.tensor(idx))
        print('%4s'%(mytokenizer.index_word[idx+1]), tmpvec.detach().numpy())

 
if __name__ == '__main__':
    dm02_nnembeding_show()
