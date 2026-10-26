import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

import joblib
# from tensorflow.keras.preprocessing.text import Tokenizer   # 词汇映射器
from keras.src.legacy.preprocessing.text import Tokenizer   # 上面的替代方案

"""
    获得词向量的one-hot方式总结：
        优点：实现、理解简单
        缺点：
            1- 词向量是一个稀疏向量，会浪费存储和计算资源
            2- 对多义词的处理不好。例如：过过过过过过，中的3对过的词向量是同一个
"""

def demo01():
    # 1- 准备语料库
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}

    # 2- 训练词汇映射器
    # 2.1- 创建映射器对象
    my_tokenizer = Tokenizer()
    # 2.2- 训练模型
    my_tokenizer.fit_on_texts(vocabs)
    # 2.3- 得到训练好的结果数据：词和索引的映射关系
    # 注意：1- word_index后面不要加小括号；2- 返回结果类型是普通的Python字典
    word_dict = my_tokenizer.word_index
    # print(type(word_dict))
    # print(word_dict)

    # 3- 得到每个词的one-hot词向量
    vocabs_len = len(vocabs)
    for word in vocabs:
        # 3.1- 初始化一个全0的词向量列表
        word_one_hot = [0]*vocabs_len   # 得到是[0,0,0,0,0,0]
        # 3.2- 获得词在词汇映射器中的索引。注意：索引是从1开始的
        index = word_dict[word]-1
        # 3.3- 将指定位置的值改为1即可
        word_one_hot[index] = 1

        print(f"{word}，对应的词向量是{word_one_hot}")

    # 4- 保存训练好的模型
    joblib.dump(my_tokenizer,"../model/tokenizer.pkl")

def demo02():
    word = "王力宏"

    # 1- 加载训练好的模型
    my_tokenizer = joblib.load("../model/tokenizer.pkl")

    # 2- 得到某个词的词向量
    # 2.1- 获得词汇映射器中词和索引的对应关系
    word_dict = my_tokenizer.word_index
    # 2.2- 初始化一个全0的列表
    word_one_hot = [0]*len(word_dict)
    # 2.3- 获得该词的索引
    index = word_dict[word]-1
    # 2.4- 将指定索引位置的值设置为1即可
    word_one_hot[index] = 1

    print(word_one_hot)

# 基于纯Python入门试学班的代码实现
def demo03():
    # 1- 准备语料库
    vocabs = ["周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"]

    # 2- 得到每个词的one-hot词向量
    vocabs_len = len(vocabs)
    for word in vocabs:
        # 2.1- 初始化一个全0的词向量列表
        word_one_hot = [0] * vocabs_len  # 得到是[0,0,0,0,0,0]
        # 2.2- 获得词在语料库的索引
        index = vocabs.index(word)
        # 3.3- 将指定位置的值改为1即可
        word_one_hot[index] = 1

        print(f"{word}，对应的词向量是{word_one_hot}")


if __name__ == '__main__':
    # 1- 训练模型：也就是训练词汇映射器
    demo01()

    # 2- 使用训练好的模型
    demo02()

    # 3- 用最基本的方式实现
    demo03()