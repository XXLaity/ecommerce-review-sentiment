# -*- coding: utf-8 -*-
"""
train.py
电商评论智能情感分析系统 —— 训练主脚本

特性：
- 中文文本预处理：jieba分词、去停用词、特殊符号过滤、短文本清洗
- TF-IDF 文本特征提取
- 双模型对比：朴素贝叶斯(MultinomialNB) + 逻辑回归(LogisticRegression)
- 5折交叉验证评估模型泛化能力
- 混淆矩阵、分类报告、情感分布饼图、模型准确率对比图
- 自动保存最佳模型（TF-IDF向量器 + 分类器）

用法：
    python train.py
    python train.py --max-features 5000 --ngram-range 2
"""
import argparse
import json
import re
import time
from pathlib import Path

import jieba
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--stopwords", type=str, default="stopwords.txt")
    p.add_argument("--max-features", type=int, default=5000, help="TF-IDF最大特征数")
    p.add_argument("--ngram-range", type=int, default=2, help="最大n-gram（1=unigram, 2=unigram+bigram）")
    p.add_argument("--cv", type=int, default=5, help="交叉验证折数")
    p.add_argument("--out", type=str, default="outputs")
    p.add_argument("--save-model", type=str, default="models/sentiment_best.joblib")
    return p.parse_args()


def load_stopwords(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def clean_text(text: str) -> str:
    """去除特殊符号、URL、数字等噪声，保留中文和英文字母"""
    text = str(text)
    text = re.sub(r"https?://\S+", "", text)          # 去URL
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", text)  # 保留中英文字母
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, stopwords: set) -> str:
    """jieba分词 + 去停用词，返回空格分隔的词语字符串"""
    words = jieba.lcut(text)
    words = [w for w in words if w.strip() and w not in stopwords and len(w.strip()) > 1]
    return " ".join(words)


def preprocess(df: pd.DataFrame, stopwords: set) -> pd.DataFrame:
    """对数据集执行完整文本预处理流水线"""
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    df["text_cut"] = df["text_clean"].apply(lambda x: tokenize(x, stopwords))
    # 过滤空文本
    df = df[df["text_cut"].str.len() > 0].reset_index(drop=True)
    return df


def plot_sentiment_dist(df: pd.DataFrame, out_dir: Path):
    """绘制情感分布饼图"""
    counts = df["label"].value_counts().sort_index()
    labels = ["负面", "正面"]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#e74c3c", "#2ecc71"]
    ax.pie(counts.values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90,
           textprops={"fontsize": 13})
    ax.set_title("评论情感分布", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "sentiment_distribution.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(results: dict, out_dir: Path):
    """绘制模型准确率对比柱状图"""
    models = list(results.keys())
    cv_acc = [results[m]["cv_mean_acc"] * 100 for m in models]
    test_acc = [results[m]["test_acc"] * 100 for m in models]
    test_f1 = [results[m]["test_f1"] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, cv_acc, width, label="5折交叉验证准确率", color="#3498db")
    ax.bar(x, test_acc, width, label="验证集准确率", color="#2ecc71")
    ax.bar(x + width, test_f1, width, label="验证集F1分数", color="#f39c12")
    ax.set_ylabel("分数 (%)")
    ax.set_title("模型性能对比", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(60, 100)
    ax.grid(axis="y", alpha=0.3)
    for i, (c, t, f) in enumerate(zip(cv_acc, test_acc, test_f1)):
        ax.text(i - width, c + 0.3, f"{c:.1f}", ha="center", fontsize=9)
        ax.text(i, t + 0.3, f"{t:.1f}", ha="center", fontsize=9)
        ax.text(i + width, f + 0.3, f"{f:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_dir: Path):
    """绘制混淆矩阵"""
    labels = ["负面", "正面"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(2)); ax.set_xticklabels(labels)
    ax.set_yticks(range(2)); ax.set_yticklabels(labels)
    ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title("混淆矩阵（逻辑回归）")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)

    # 加载停用词
    stopwords = load_stopwords(args.stopwords)
    print(f"停用词数: {len(stopwords)}")

    # 加载并预处理数据
    train_df = pd.read_csv(Path(args.data) / "train.csv")
    val_df = pd.read_csv(Path(args.data) / "val.csv")
    print(f"原始训练集 {len(train_df)} 条，验证集 {len(val_df)} 条")

    print("正在进行文本预处理（分词+去停用词）...")
    t0 = time.time()
    train_df = preprocess(train_df, stopwords)
    val_df = preprocess(val_df, stopwords)
    print(f"预处理完成，耗时 {time.time()-t0:.1f}s | 训练集 {len(train_df)} 条，验证集 {len(val_df)} 条")

    # 情感分布可视化
    plot_sentiment_dist(train_df, out_dir)

    # TF-IDF 特征提取
    ngram = (1, args.ngram_range)
    tfidf = TfidfVectorizer(max_features=args.max_features, ngram_range=ngram, sublinear_tf=True)
    X_train = tfidf.fit_transform(train_df["text_cut"])
    y_train = train_df["label"].values
    X_val = tfidf.transform(val_df["text_cut"])
    y_val = val_df["label"].values
    print(f"TF-IDF特征矩阵: 训练 {X_train.shape}, 验证 {X_val.shape}")

    # 定义模型
    models = {
        "朴素贝叶斯": MultinomialNB(alpha=1.0),
        "逻辑回归": LogisticRegression(max_iter=1000, C=1.0, solver="liblinear"),
    }

    results = {}
    best_model_name, best_f1 = None, 0.0

    for name, clf in models.items():
        print(f"\n=== {name} ===")
        # 5折交叉验证
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        print(f"  {args.cv}折交叉验证准确率: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")

        # 在完整训练集上训练，验证集测试
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        test_acc = accuracy_score(y_val, y_pred)
        test_f1 = f1_score(y_val, y_pred, average="weighted")
        print(f"  验证集准确率: {test_acc*100:.2f}% | F1: {test_f1*100:.2f}%")

        results[name] = {
            "cv_mean_acc": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "test_acc": round(test_acc, 4),
            "test_f1": round(test_f1, 4),
        }

        if name == "逻辑回归":
            print(f"\n{name} 分类报告:")
            print(classification_report(y_val, y_pred, target_names=["负面", "正面"], digits=4))
            cm = confusion_matrix(y_val, y_pred)
            plot_confusion(cm, out_dir)

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_name = name

    # 模型对比图
    plot_model_comparison(results, out_dir)

    # 保存最佳模型（TF-IDF + 分类器 + 元信息）
    best_clf = models[best_model_name]
    best_clf.fit(X_train, y_train)  # 确保在完整训练集上
    joblib.dump({
        "tfidf": tfidf,
        "model": best_clf,
        "model_name": best_model_name,
        "stopwords": stopwords,
        "class_names": ["负面", "正面"],
        "max_features": args.max_features,
        "ngram_range": ngram,
    }, args.save_model)

    summary = {
        "dataset": "ChnSentiCorp 中文在线评论情感分析数据集",
        "train_size": len(train_df),
        "val_size": len(val_df),
        "tfidf_features": X_train.shape[1],
        "best_model": best_model_name,
        "best_val_acc": results[best_model_name]["test_acc"],
        "best_val_f1": results[best_model_name]["test_f1"],
        "models": results,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"最佳模型: {best_model_name}")
    print(f"验证集准确率: {results[best_model_name]['test_acc']*100:.2f}%")
    print(f"验证集F1: {results[best_model_name]['test_f1']*100:.2f}%")
    print(f"模型已保存: {args.save_model}")
    print(f"输出文件: {out_dir}/")


if __name__ == "__main__":
    main()
