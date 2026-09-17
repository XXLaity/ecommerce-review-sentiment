# 基于多算法融合的电商评论智能情感分析系统

基于自然语言处理与机器学习技术的中文评论文本情感分类系统，自动判别评论的正面/负面情感，支持单条与批量评论预测，输出情感类别与置信度，可辅助商家进行口碑统计与舆情分析。

## 项目简介

- 针对电商行业海量用户评论人工统计效率低、舆情反馈滞后的痛点，采用 TF-IDF 文本特征提取 + 机器学习分类器，实现中文评论的自动情感二分类（正面/负面）
- 使用公开 ChnSentiCorp 中文在线评论情感分析数据集（含酒店、电子产品、书籍等评论，共 12000 条，正负样本完全均衡）
- 对比朴素贝叶斯与逻辑回归两种机器学习模型，使用 5 折交叉验证评估泛化能力，逻辑回归模型验证集准确率达到 **89.1%**
- 完整中文文本预处理流水线：特殊符号过滤、jieba 中文分词、停用词剔除、TF-IDF 特征向量化
- 输出情感分布饼图、模型性能对比图、混淆矩阵等可视化分析结果
- 支持单条评论即时预测与批量文件预测，输出情感类别与置信度

## 技术栈

Python · Scikit-learn · Jieba · TF-IDF · Pandas · NumPy · Matplotlib · Joblib

## 项目结构

```
project2_sentiment/
├── prepare_data.py        # 数据集准备（arrow→CSV，80/20分层划分）
├── train.py               # 训练主脚本（预处理+TF-IDF+NB+LR+交叉验证+可视化）
├── predict.py             # 推理脚本（单条/批量/演示模式）
├── stopwords.txt          # 中文停用词表
├── requirements.txt       # 依赖清单
├── data/                  # 划分后的数据（train.csv / val.csv）
├── data_raw/              # 原始 arrow 数据集
├── models/
│   └── sentiment_best.joblib   # 最佳模型（TF-IDF+分类器+停用词）
└── outputs/
    ├── sentiment_distribution.png   # 情感分布饼图
    ├── model_comparison.png         # 模型性能对比图
    ├── confusion_matrix.png         # 混淆矩阵
    ├── summary.json                 # 训练总结
    └── predict/                     # 推理结果
```

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

从 Hugging Face 下载公开 ChnSentiCorp 数据集（3个arrow文件，约3.8MB）：

```bash
mkdir -p data_raw
for split in train validation test; do
  curl -L -o "data_raw/chnsenticorp-${split}.arrow" \
    "https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/chn_senti_corp-${split}.arrow"
done
python prepare_data.py
```

### 3. 训练模型

```bash
python train.py
```

训练完成后输出：最佳模型 `models/sentiment_best.joblib`、情感分布饼图、模型对比图、混淆矩阵、分类报告与 `outputs/summary.json`。

### 4. 推理预测

```bash
# 单条评论
python predict.py --text "这个商品质量很好，非常满意"

# 批量文件（每行一条评论）
python predict.py --file comments.txt

# 演示模式（不传参数自动运行示例）
python predict.py
```

## 实验结果

> 以下为真实训练输出（ChnSentiCorp 数据集，80%训练/20%验证，正负样本均衡）。

| 指标 | 朴素贝叶斯 | 逻辑回归（最佳） |
| --- | --- | --- |
| 5折交叉验证准确率 | 86.24% ± 0.79% | 87.27% ± 0.96% |
| 验证集准确率 | 88.25% | **89.08%** |
| 验证集F1分数 | 88.25% | **89.08%** |

**逻辑回归分类报告（验证集 2399 条）：**

| 类别 | precision | recall | f1-score | 样本数 |
| --- | --- | --- | --- | --- |
| 负面 | 87.9% | 90.6% | 89.2% | 1199 |
| 正面 | 90.3% | 87.6% | 88.9% | 1200 |

- 训练集：9598 条（过滤空文本后），验证集：2399 条
- TF-IDF 特征数：5000（unigram + bigram）
- 训练耗时：约 5 秒（CPU）

## 面试常见问题速答

**Q1：为什么选用 TF-IDF？**
A：TF-IDF 衡量词语在文本中的重要程度（词频×逆文档频率），能把文本转为计算机可识别的向量特征，是传统文本分类的标准特征工程方法，计算高效、可解释性强。

**Q2：朴素贝叶斯和逻辑回归哪个效果好？为什么？**
A：在这个数据集上逻辑回归略优（89.1% vs 88.2%）。朴素贝叶斯基于贝叶斯公式，假设特征条件独立，训练速度极快；逻辑回归是判别式模型，直接学习分类边界，对特征独立性没有强假设，文本场景通常效果更稳定。

**Q3：怎么处理中文文本？**
A：先用正则去除URL、特殊符号、数字等噪声，保留中英文字母；再用jieba进行中文分词；然后过滤停用词和单字词；最后用TF-IDF将分词结果转为特征向量。

**Q4：模型过拟合了吗？**
A：没有。5折交叉验证准确率（87.3%）和验证集准确率（89.1%）接近，交叉验证标准差仅0.96%，说明模型泛化能力稳定，没有明显过拟合。

**Q5：还能怎么提升准确率？**
A：可以尝试：增大TF-IDF特征数、加入三gram特征、使用SVM或XGBoost、引入预训练词向量（Word2Vec/BERT）做文本表示、处理网络用语和表情符号等。

## License

数据集来源：ChnSentiCorp (https://huggingface.co/datasets/seamew/ChnSentiCorp)，仅供学习研究使用。
