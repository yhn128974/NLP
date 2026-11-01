import jieba
from itertools import chain
import pandas as pd
from pathlib import Path


dir_path = Path(__file__).resolve().parent.parent

path=dir_path/'data'/'train.tsv'

# print(f'BASE_DIR:{BASE_DIR}')
if __name__ == '__main__':
    df = pd.read_csv(path,sep="\t",encoding="UTF-8")

    # map_result = map(lambda line:jieba.lcut(line), df["sentence"])
    # word_set = set(chain(*map_result))

    # 合并的写法版本
    word_set = set(chain(*map(lambda line:jieba.lcut(line), df["sentence"])))
    print("词汇总个数",len(word_set))

    # name_list=['zhangsan','hello','world!']
    # result_name= list(map(lambda name:len(name),name_list))
    # print(result_name)

    
 