import jieba.posseg as pseg

if __name__ == '__main__':
    content = "我爱自然语言处理"

    # [pair('我', 'r'), pair('爱', 'v'), pair('自然语言', 'l'), pair('处理', 'v')]。将pair当成元组使用即可
    # pseg是分词加词性标注的模块，与jieba的区别是返回的是pair，而jieba返回的是字符串
    result = pseg.lcut(content)
    print(result)

    for word,pos in result:
        print(f"{word} 的词性是 {pos}")