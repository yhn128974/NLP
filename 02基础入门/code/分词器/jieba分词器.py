import jieba

"""
    掌握的内容：result = jieba.lcut(content)
"""

content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

def demo01():
  
    result = jieba.lcut(content)
    # result = list(jieba.cut(content))

    # ['传智', '教育', '是', '一家', '上市公司', '，', '旗下', '有', '黑马', '程序员', '品牌', '。', '我', '是', '在', '黑马', '这里', '学习', '人工智能']
    print(result)
    print(type(result)) # list

def demo02():
    # 全词模式
    result = jieba.lcut(content,cut_all=True)
    print(result)

def demo03():
    # 搜索引擎模式：给搜索引擎使用的模式
    result = jieba.lcut_for_search(content)
    print(result)

if __name__ == '__main__':
    # 精确模式
    demo01()

    # 全词模式
    demo02()

    # 搜索引擎模式：给搜索引擎使用的模式
    demo03()