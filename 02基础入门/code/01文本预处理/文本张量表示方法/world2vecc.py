from pathlib import Path
import shutil
import fasttext

# 相对路径是相对“运行时的工作目录”，不是相对本脚本；用 __file__ 锚定脚本目录
BASE_DIR = Path(__file__).resolve().parent
DATA_SRC = BASE_DIR.parent / "data" / "gz03ag"
MODEL_DST = BASE_DIR.parent / "model" / "word2vec.bin"

# fastText 在 Windows 下打不开含中文的路径，训练/读写都放到纯英文缓存目录
CACHE_DIR = Path.home() / ".cache" / "nlp_ft_demo"
DATA_PATH = CACHE_DIR / "gz03ag.txt"
MODEL_PATH = CACHE_DIR / "word2vec.bin"


def _ensure_ascii_data() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_SRC.exists():
        raise FileNotFoundError(f"找不到训练数据: {DATA_SRC}")
    need_copy = (
        not DATA_PATH.exists()
        or DATA_PATH.stat().st_size != DATA_SRC.stat().st_size
    )
    if need_copy:
        shutil.copy2(DATA_SRC, DATA_PATH)


def demo01():
    # 该语料没有 __label__，适合无监督学词向量（skipgram）
    _ensure_ascii_data()
    # 无监督学习
    # 参数含义：str(DATA_PATH) 数据路径，model="skipgram" 模型类型，dim=100 词向量维度，epoch=1 训练轮数，minCount=5 最小词频
    # 返回值：model 模型对象
    model = fasttext.train_unsupervised(
        str(DATA_PATH),
        model="skipgram",#可以切换为CBOW模型
        # model="cbow",
        dim=100,
        epoch=1,
        minCount=5,
    )

    # 保存到英文路径，再复制回项目 model 目录（Python 可写中文路径）
    model.save_model(str(MODEL_PATH))
    MODEL_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_PATH, MODEL_DST)
    print(f"model saved: {MODEL_DST}")


def demo02():
    # # 从英文缓存加载（fastText 同样读不了中文路径）
    # if not MODEL_PATH.exists():
    #     if MODEL_DST.exists():
    #         CACHE_DIR.mkdir(parents=True, exist_ok=True)
    #         shutil.copy2(MODEL_DST, MODEL_PATH)
    #     else:
    #         raise FileNotFoundError("请先运行 demo01() 训练模型")

    model = fasttext.load_model(str(MODEL_PATH))
    word_vector = model.get_word_vector("city")
    print(f"word_vector:{word_vector}")

def demo03():
    model = fasttext.load_model(str(MODEL_PATH))
    # 获得相近的词
    similar_words = model.get_nearest_neighbors("city")
    print(f"similar_words:{similar_words}")




if __name__ == "__main__":
    # demo01()
    # demo02()
    demo03()
