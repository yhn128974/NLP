import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

# 下面两个等价
# from tensorflow.keras.preprocessing import sequence
from keras.preprocessing import sequence

"""
    句子长度规范的作用【掌握】：
        1- Dataloader(batch_size=1)，张量的形状不相同的时候只能设置为1。因为里面是使用stack对张量进行拼接
            stack的要求是张量形状完全相同
        2- 迁移学习中有现成的参数可以进行设置
"""

def demo01():
    # 1- 准备数据，两条句子
    sen_list = [
        [1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
        [2, 32, 1, 23, 1]
    ]

    # 2- 句子长度处理
    """
        参数解释：
            sequences：要进行句子长度规范处理的句子列表
            maxlen：句子长度规范的上限。也就是句子中词个数的上限
            padding：对少于长度上限的进行填充。推荐用post
                pre：在句子的前面进行填充
                post：在句子的后面进行填充
            truncating：对大于长度上限的进行截断。推荐用post
                pre：在句子的前面进行截断
                post：在句子的后面进行截断
            value：padding的时候的填充值。推荐使用0，因为对句子的含义不会造成影响
    """
    # result = sequence.pad_sequences(sequences=sen_list,maxlen=10)
    # pad_sequences: 句子长度规范
    result = sequence.pad_sequences(sequences=sen_list,maxlen=10,padding="post",truncating="post",value=6666)
    print(result) 
 
def demo02():
    # 1- 准备数据
    sen_list = [
        [1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
        [2, 32, 1, 23, 1]
    ]

    # 2- 分别对每条句子进行长度规范的处理
    result_list = []
    for sen in sen_list:
        if len(sen)>10:
            # 截断
            result_list.append(sen[:10])
        else:
            result_list.append(sen + [0] * (10 - len(sen)))

    print(result_list)

if __name__ == '__main__':
    # demo01()

    demo02()