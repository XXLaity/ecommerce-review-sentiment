# -*- coding: utf-8 -*-
"""
predict.py
电商评论情感分析推理脚本：加载训练好的模型，对单条或批量评论进行情感预测，输出正面/负面及置信度。

用法：
    python predict.py --text "这个商品质量很好，非常满意"
    python predict.py --file 评论文件.txt
"""
import argparse
import json
import re
from pathlib import Path

import jieba
import joblib


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, stopwords: set) -> str:
    words = jieba.lcut(text)
    words = [w for w in words if w.strip() and w not in stopwords and len(w.strip()) > 1]
    return " ".join(words)


def predict_sentiment(text: str, bundle: dict) -> dict:
    tfidf = bundle["tfidf"]
    model = bundle["model"]
    stopwords = bundle["stopwords"]
    class_names = bundle["class_names"]

    cleaned = clean_text(text)
    cut = tokenize(cleaned, stopwords)
    if not cut:
        return {"label": "未知", "confidence": 0.0, "probabilities": {}}
    x = tfidf.transform([cut])
    proba = model.predict_proba(x)[0]
    idx = int(proba.argmax())
    return {
        "label": class_names[idx],
        "confidence": round(float(proba[idx]), 4),
        "probabilities": {class_names[i]: round(float(proba[i]), 4) for i in range(len(class_names))},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default=None, help="单条评论文本")
    p.add_argument("--file", type=str, default=None, help="包含多条评论的txt文件（每行一条）")
    p.add_argument("--model", type=str, default="models/sentiment_best.joblib")
    p.add_argument("--out", type=str, default="outputs/predict")
    args = p.parse_args()

    bundle = joblib.load(args.model)
    print(f"模型加载完成: {bundle['model_name']}，类别: {bundle['class_names']}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    if args.text:
        r = predict_sentiment(args.text, bundle)
        print(f"\n评论: {args.text}")
        print(f"情感: {r['label']} (置信度 {r['confidence']*100:.1f}%)")
        print(f"各类概率: {r['probabilities']}")
        results.append({"text": args.text, **r})

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"\n共读取 {len(lines)} 条评论:")
        for line in lines:
            r = predict_sentiment(line, bundle)
            print(f"  [{r['label']}] ({r['confidence']*100:.0f}%) {line[:50]}")
            results.append({"text": line, **r})

    else:
        # 演示模式：用几条示例评论
        demos = [
            "这个商品质量很好，发货速度快，非常满意，下次还会买",
            "东西太差了，用了两天就坏了，客服态度也不好，差评",
            "物流挺快的，包装也还行，东西一般般吧，没有想象中好",
            "性价比很高，推荐购买，五星好评",
            "完全不值这个价，质量差，退货了",
        ]
        print("\n=== 演示模式（示例评论）===")
        for d in demos:
            r = predict_sentiment(d, bundle)
            print(f"  [{r['label']}] ({r['confidence']*100:.0f}%) {d}")
            results.append({"text": d, **r})

    with open(out_dir / "predict_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {out_dir}/predict_results.json")


if __name__ == "__main__":
    main()
