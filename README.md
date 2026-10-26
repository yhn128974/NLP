# 个人学习周报

**周期**：2026年10月19日 至 2026年10月26日
**学习主题**：深度学习全模块通关（CNN & RNN）与 Word2Vec 开山论文精读

---

### 一、 上周核心学习成果

1. **卷积神经网络（CNN）入门与图像分类案例**：
   - 掌握图像多通道张量表示与 CNN 拓扑构成，深入理解卷积层的局部感受野与权值共享机制，推导 Padding、Stride 与输出特征图尺寸计算。
   - 掌握最大池化与平均池化的下采样降维特性及全连接层分类映射，基于 PyTorch 搭建经典 CNN 架构并完成 CIFAR-10 图像分类案例实战。

2.**循环神经网络（RNN）入门与AI歌词生成器案例**：

- **词嵌入与循环层**：理解词嵌入层（`nn.Embedding`）将离散词索引映射为稠密向量的原理与 API 演示[cite: 20, 21]；掌握 RNN 循环网络层隐状态传递机制（$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$）与 API 调用[cite: 20, 21, 22]。
- **RNN_AI歌词生成器案例**：从零完成歌词文本词表（Vocab）构建、时序滑动窗口数据集封装、包含 Embedding 与 RNN 层的生成网络搭建，规范编写模型训练与歌词续写生成测试全链路代码[cite: 21]。

  3.**前沿文献精读：Word2Vec 开山论文**：s

- 精读 Mikolov et al. (2013) 《Efficient Estimation of Word Representations in Vector Space》。

---

### 二、 本周学习计划

**正式迈入 NLP 自然语言处理核心与进阶阶段：**

1. **文本特征工程与词向量训练**：掌握 jieba 分词、n-gram 特征、文本长度规范与回译数据增强，基于 FastText 完成 Word2Vec 词向量训练与快速文本分类。
2. **门控序列模型与注意力机制**：深入学习 LSTM、Bi-LSTM 与 GRU 门控结构并完成人名分类器案例；攻克 Seq2Seq 架构与 Attention 注意力机制原理（结合精读 **Bahdanau 2014 Attention 论文**），完成**英译法机器翻译**实战。
3. 论文：注意力机制与序列建模范式Seq2Seq + Attention (Bahdanau et al., 2014)
