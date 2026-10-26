import jieba

"""
    总结：全词、精确、搜索引擎模式的区别是分词结果的词的个数不同
    掌握：jieba.lcut(sentence=content)
"""

content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

def demo01():
    # 推荐使用：返回结果类型是List列表
    result = jieba.lcut(sentence=content,cut_all=False)
    print(type(result))
    print(result)

    print("-"*30)

    # 不带l的返回结果类型是generator生成器。节约内存资源
    result = jieba.cut(sentence=content, cut_all=False)
    print(type(result))
    print(result)
    print(next(result))

def demo02():
    # 全词的模式：cut_all=True。和上面的区别是该方式分的词会更加的精细
    result = jieba.lcut(sentence=content, cut_all=True)
    print(result)

def demo03():
    result = jieba.lcut_for_search(sentence=content)
    print(result)

if __name__ == '__main__':
    # 【推荐】精确模式
    demo01()

    # 全词模式
    demo02()

    # 搜索引擎模式
    demo03()
