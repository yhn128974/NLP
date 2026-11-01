"""
one_hot ：实现简单，但是浪费内存，和CPU资源
对一词多义的理解能力很差
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
import jieba
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
# 或者使用: from keras.src.legacy.preprocessing.text import Tokenizer

current_file_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_file_path, "models")
os.makedirs(file_path, exist_ok=True)




# 自己实现ONEHOT
def demo01():

    # 数据准备
    vocabs={'周杰伦','蔡依林','王力宏','林俊杰','张学友'}
    
    # 创建词汇映射器
    my_tokenizer=Tokenizer()
    # 训练词汇映射器
    my_tokenizer.fit_on_texts(vocabs)
    # 获取映射关系
    word_dict=my_tokenizer.word_index

    # 保存训练好的词汇映射器
    joblib.dump(my_tokenizer, os.path.join(file_path, 'my_tokenizer.pkl'))

    print(word_dict)

    # 构建独热模式
    for word in vocabs:
        word_one_hot=[0]*len(vocabs)
        index=word_dict[word]-1
        word_one_hot[index]=1
        print(f'{word}:{word_one_hot}')
        
 
# 使用预训练的词分类器
def demo02():

    # 数据准备
    start_name='周杰伦'

    # 加载预训练的词分类器
    my_tokenizer = joblib.load(os.path.join(file_path, 'my_tokenizer.pkl'))

     # 获取映射关系
    word_dict=my_tokenizer.word_index

    word_one_hot=[0]*len(word_dict)

    if start_name not in word_dict:
        print(f'{start_name} 不在词表中: {list(word_dict)}')
        return
    index=word_dict[start_name]-1

    word_one_hot[index]=1

    print(f'{start_name}:{word_one_hot}')
    

# 使用数组方式实现
def demo03():
      # 数据准备
    vocabs=['周杰伦','蔡依林','王力宏','林俊杰','张学友']

    for vocab in vocabs:
        word_one_hot=[0]*len(vocabs)
        index=vocabs.index(vocab)
        word_one_hot[index]=1
        print(f'{vocab} one_hot向量{word_one_hot}')


if __name__ == '__main__':
    # demo01()
    # demo02()
    demo03()
   