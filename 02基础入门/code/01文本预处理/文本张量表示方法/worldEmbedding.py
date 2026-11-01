import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 数字必须变成字符串
import torch
import torch.nn as nn
import jieba
from keras.src.legacy.preprocessing.text import Tokenizer
# 词向量可视化工具
from torch.utils.tensorboard import SummaryWriter

# Windows + 中文路径时，补齐 tensorboard 对 gfile 的兼容
import tensorflow as tf
import tensorboard.compat.tensorflow_stub.io.gfile as tb_gfile

tb_gfile.join = os.path.join
tf.io.gfile = tb_gfile


if __name__ == "__main__":
    # 1- 准备文本内容
    sentence1 = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    sentence2 = "我爱自然语言处理"
    sentence_list = [sentence1, sentence2]

    # 2- 分词处理
    word_list = []
    for sentence in sentence_list:
        word_list.append(jieba.lcut(sentence))

    # print(f'word_list:{word_list}')

    # 3.训练词汇映射表
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(word_list)
    index_word = my_tokenizer.index_word
    print(f"index_word:{index_word}")

    # 4.创建词嵌入层
    # 参数解释：
    # num_embeddings：词汇表大小，
    # embedding_dim：词向量维度，一般是2的N次方
    embeddinger = nn.Embedding(num_embeddings=len(index_word), embedding_dim=4)

    # 获取每个词的词向量
    for index in range(len(index_word)):
        word_vector = embeddinger(torch.tensor(index))
        print(f"词索引{index + 1} ,词内容：{index_word[index + 1]}，词向量：{word_vector}")



    # 目标路径
    log_dir = Path(__file__).resolve().parent.parent / "runs" #目录有中文就会报错
    log_dir.mkdir(parents=True, exist_ok=True)

    # 可视化：生成到 01文本预处理/runs
    summary_writer = SummaryWriter(log_dir=str(log_dir))
    summary_writer.add_embedding(embeddinger.weight.data, list(index_word.values()))
    summary_writer.close()

    print(f"日志目录: {log_dir}")
    print(f"tensorboard --logdir=\"{log_dir}\"")
