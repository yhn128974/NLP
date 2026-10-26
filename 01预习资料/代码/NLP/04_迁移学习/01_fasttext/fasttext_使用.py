# pip install fasttext-wheel

import fasttext

def demo01():
    # 1- 模型训练
    model = fasttext.train_supervised(input="../data/cooking_train.txt")

    # 2- 模型预测
    # 参数解释：k指的是返回top K个预测概率最高的目标值
    pred_labels = model.predict(text="Which baking dish is best to bake a banana bread ?",k=3)
    print(type(pred_labels))    # 类型是嵌套元组
    print(pred_labels)

    pred_labels = model.predict(text="how to seperate peanut oil from roasted peanuts at home?",k=5)
    print(pred_labels)

    # 3- 模型评估
    result = model.test(path="../data/cooking_valid.txt")
    print(result)

# 将字母全部统一成小写，而且标点符号的前面增加空格
def demo02():
    # 训练模型
    model = fasttext.train_supervised(input="../data/cooking.pre.train")

    # 模型评估
    result = model.test(path="../data/cooking.pre.valid")
    print(result)

# 增加训练轮次
def demo03():
    # 训练模型
    model = fasttext.train_supervised(input="../data/cooking.pre.train",epoch=20)

    # 模型评估
    result = model.test(path="../data/cooking.pre.valid")
    print(result)

# 调整学习率
def demo04():
    # 训练模型
    model = fasttext.train_supervised(input="../data/cooking.pre.train",epoch=20,lr=1)

    # 模型评估
    result = model.test(path="../data/cooking.pre.valid")
    print(result)

def demo05():
    # 训练模型
    model = fasttext.train_supervised(input="../data/cooking.pre.train",epoch=20,lr=1,wordNgrams=2)

    # 模型评估
    result = model.test(path="../data/cooking.pre.valid")
    print(result)

def demo06():
    # 训练模型
    # hs：层次softmax
    model = fasttext.train_supervised(input="../data/cooking.pre.train",epoch=20,lr=1,wordNgrams=2,loss="hs")

    # 模型评估
    result = model.test(path="../data/cooking.pre.valid")
    print(result)

def demo07():
    # 训练模型
    # hs：层次softmax
    model = fasttext.train_supervised(
        input="../data/cooking.pre.train",
        autotuneDuration=60*2,
        autotuneValidationFile="../data/cooking.pre.valid"
    )

    # 模型评估
    result = model.test(path="../data/cooking.pre.valid")
    print(result)

# 将 多标签多分类 问题简化成 单标签多分类的问题。每种分类单独进行训练
def demo08():
    # 训练模型
    # ova：多标签多分类 问题简化成 单标签多分类的问题
    model = fasttext.train_supervised(input="../data/cooking.pre.train", epoch=20, lr=0.1, wordNgrams=2, loss="ova")

    # 模型评估
    result = model.test(path="../data/cooking.pre.valid")
    print(result)

def demo09():
    # 训练模型
    # ova：多标签多分类 问题简化成 单标签多分类的问题
    model = fasttext.train_supervised(input="../data/cooking.pre.train", epoch=20, lr=0.1, wordNgrams=2, loss="ova")

    # 保存训练好的模型
    model.save_model("../model/fasttext.pkl")

    # 加载训练好的模型
    model = fasttext.load_model("../model/fasttext.pkl")
    pred_labels = model.predict(text="how to seperate peanut oil from roasted peanuts at home?", k=5)
    print(pred_labels)


if __name__ == '__main__':
    # 1- 模型训练、预测、评估
    # demo01()    # (3000, 0.15066666666666667, 0.06515784921435779)

    # 2- 数据基本处理
    # demo02()    # (3000, 0.17066666666666666, 0.07380712123396281)

    # 3- 增加训练轮次
    # demo03()    # (3000, 0.486, 0.2101773100764019)

    # 4- 调整学习率
    # demo04()    # (3000, 0.5813333333333334, 0.2514055067031858)

    # 5- 设置n-gram参数
    # demo05()    # (3000, 0.5926666666666667, 0.25630676084762866)

    # 6- 调整损失函数：主要是提升了运行速度
    # demo06()      # (3000, 0.597, 0.25818076978520976)

    # 7- 自动超参数调优
    # demo07()    # (3000, 0.53, 0.22920570851953295)

    # 8- 多标签多分类问题
    # demo08()    # (3000, 0.5286666666666666, 0.2286290903848926)

    # 9- 保存模型和重新加载模型
    demo09()