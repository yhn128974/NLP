import os

import torch

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

from transformers import pipeline   # 管道形式

def text_classification():
    # 1- 加载预训练模型
    """
        参数解释：
            task：业务的任务类型。这里是文本分类，可以传递text-classification或sentiment-analysis，推荐使用text-classification
                注意：其他任务的task值去pipeline源代码中找
            model：预训练模型的路径。推荐前面加上r，避免转义
    """
    model = pipeline(task="text-classification",model=r"D:\soft\PretrainedModel\chinese_sentiment")   # 5分类问题
    # model = pipeline(task="text-classification",model=r"D:\soft\PretrainedModel\bert-base-chinese")     # 二分类。好评是1，差评是0

    # 2- 预测
    # pred_result = model("我爱北京天安门，天安门上太阳升。")
    pred_result = model("这家餐馆的卫生太差了，吃了拉稀，非常不推荐")
    # pred_result = model("这家餐馆的卫生还行，就是有点油")

    print(type(pred_result))
    print(pred_result)  # [{'label': 'star 5', 'score': 0.6314295530319214}]

def text_feature_extraction():
    # 1- 加载预训练模型
    model = pipeline(task="feature-extraction",model=r"D:\soft\PretrainedModel\bert-base-chinese")

    # 2- 文本特征提取：先分词->以列表形式返回词向量
    # 返回的形状是 [1, 17, 768]。1是句子条数，17是因为对每个字进行分词加上句子的开始和结束；768词向量的维度数
    result = model("这家餐馆的卫生还行，就是有点油")

    print(type(result))
    print(result)

    print(torch.tensor(result).shape)

def fill_blank():
    # 1- 加载预训练模型
    # model = pipeline(task="fill-mask",model=r"D:\soft\PretrainedModel\bert-base-chinese")
    model = pipeline(task="fill-mask",model=r"D:\soft\PretrainedModel\chinese-bert-wwm")

    # 2- 填空
    """
        注意：要进行填充的地方，必须写 [MASK]
    """
    content = "我想明天去[MASK]家吃饭。"
    result = model(content)
    print(type(result))
    print(result)

def q_and_a():
    # 1- 加载模型
    model = pipeline(task="question-answering",model=r"D:\soft\PretrainedModel\chinese_pretrain_mrc_roberta_wwm_ext_large")

    # 因为bert-base-chinese模型训练的语料库全都是中文的，因此对英文内容处理不好
    # model = pipeline(task="question-answering",model=r"D:\soft\PretrainedModel\bert-base-chinese")

    # 2- 提问
    # context = '我叫张三，我是一个程序员，我的喜好是打篮球。'
    # context = '张三是我的名字，程序员是我的职业，打篮球是我的喜好。'
    context = 'my name is zhangsan，my job is programmer，i like play basketball'
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']
    answers = model(context=context, question=questions)
    print(type(answers))
    print(answers)

def summary():
    # 1- 加载模型
    model = pipeline(task="summarization",model=r"D:\soft\PretrainedModel\distilbart-cnn-12-6")

    # 2- 提供语料库
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural " \
           "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
           "models like GPT which internally mask the future tokens. It allows the model to learn " \
           "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
           " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
           "sentences that were next to each other in the original text, sometimes not. The model then " \
           "has to predict if the two sentences were following each other or not."

    summary_result = model(text)
    print(type(summary_result))
    print(summary_result)

def ner():
    # 1- 加载模型
    model = pipeline(task="ner", model=r"D:\soft\PretrainedModel\roberta-base-finetuned-cluener2020-chinese")

    # 2- 提取实体
    print(model('鲁迅原名周树人，代表作有朝花夕拾，在商务部上班，今天他去故宫游览'))
    """
        B表示命名实体的开始，I是命名实体的中间内容
        
        {'entity': 'B-name', 'score': 0.9945884, 'index': 1, 'word': '鲁', 'start': 0, 'end': 1}, 
        {'entity': 'I-name', 'score': 0.99043053, 'index': 2, 'word': '迅', 'start': 1, 'end': 2}, 
        
        {'entity': 'B-name', 'score': 0.9791542, 'index': 5, 'word': '周', 'start': 4, 'end': 5}, 
        {'entity': 'I-name', 'score': 0.97904646, 'index': 6, 'word': '树', 'start': 5, 'end': 6}, 
        {'entity': 'I-name', 'score': 0.9797911, 'index': 7, 'word': '人', 'start': 6, 'end': 7}, 
        
        {'entity': 'I-organization', 'score': 0.3546072, 'index': 20, 'word': '务', 'start': 19, 'end': 20}, 
        {'entity': 'I-organization', 'score': 0.32793036, 'index': 21, 'word': '部', 'start': 20, 'end': 21}, 
        
        {'entity': 'B-scene', 'score': 0.881768, 'index': 29, 'word': '故', 'start': 28, 'end': 29}, 
        {'entity': 'I-scene', 'score': 0.91957027, 'index': 30, 'word': '宫', 'start': 29, 'end': 30}
    """

if __name__ == '__main__':
    # 1- 文本分类
    # text_classification()

    # 2- 文本特征提取
    # text_feature_extraction()

    # 3- 完型填空
    # fill_blank()

    # 4- 阅读理解
    # q_and_a()

    # 5- 文本摘要
    # summary()

    # 6- NER命名实体识别
    ner()
