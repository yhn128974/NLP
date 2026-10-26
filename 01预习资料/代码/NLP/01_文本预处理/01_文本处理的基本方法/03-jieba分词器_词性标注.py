import jieba.posseg as pseg

if __name__ == '__main__':
    content = "我爱北京天安门"
    result = pseg.lcut(content)
    print(result)
    print(type(result))

    for word,pos in result:
        print(f"{word}={pos}")
