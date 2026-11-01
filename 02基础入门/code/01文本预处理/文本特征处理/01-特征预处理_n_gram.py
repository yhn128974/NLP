# 引入 N-gram 处理的核心目的，是为了在特征表示中保留词序关系与局部上下文信息
def n_gram_fn(n):
    # 1- 准备数据
    word_list = ["aa", "bb", "cc", "dd"]

    # 2- 对相邻的词进行合并
    """
        2-gram：bi-gram，将相邻的2个词合并成新的词
            列表一：["aa", "bb", "cc", "dd"]
            列表二：["bb", "cc", "dd"]
            结果是：[aabb,bbcc,ccdd]
        
        3-gram：tri-gram，将相邻的3个词合并成新的词
            列表一：["aa", "bb", "cc", "dd"]
            列表二：["bb", "cc", "dd"]
            列表三：["cc", "dd"]
            结果是：[aabbcc,bbccdd]
    """
    lists = [word_list[i:] for i in range(n)]
    print(lists)

    print(f'zip:{list(zip(*lists))}')

if __name__ == '__main__':
    n_gram_fn(1)
    n_gram_fn(2)
    n_gram_fn(3)