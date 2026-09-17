# -*- coding: utf-8 -*-
"""
app.py
电商评论情感分析 - Gradio 在线交互界面
加载训练好的逻辑回归模型，对用户输入的评论进行情感二分类预测。
"""
import os
import re
import jieba
import joblib
import gradio as gr

# 加载模型
MODEL_PATH = os.getenv("MODEL_PATH", "models/sentiment_best.joblib")
bundle = joblib.load(MODEL_PATH)
tfidf = bundle["tfidf"]
model = bundle["model"]
stopwords = bundle["stopwords"]
class_names = bundle["class_names"]
model_name = bundle.get("model_name", "逻辑回归")
print(f"[模型加载] {model_name}, 类别: {class_names}")


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> str:
    words = jieba.lcut(text)
    words = [w for w in words if w.strip() and w not in stopwords and len(w.strip()) > 1]
    return " ".join(words)


def analyze(text: str):
    """对单条评论进行情感预测，返回格式化结果。"""
    if not text or not text.strip():
        return "⚠️ 请输入评论内容", "", "", ""

    cleaned = clean_text(text)
    cut = tokenize(cleaned)

    if not cut:
        return "⚠️ 文本清洗后无有效内容", "", "", ""

    x = tfidf.transform([cut])
    proba = model.predict_proba(x)[0]
    idx = int(proba.argmax())
    label = class_names[idx]
    confidence = float(proba[idx])

    # 标签展示
    emoji = "😊 正面" if label == "正面" else "😞 负面"
    label_html = f"<div style='text-align:center;font-size:28px;font-weight:bold;padding:16px;background:{'#e8f5e9' if label=='正面' else '#ffebee'};border-radius:12px;color:{'#2e7d32' if label=='正面' else '#c62828'};'>{emoji}</div>"

    # 置信度
    conf_html = f"<div style='text-align:center;font-size:20px;margin-top:8px;'>置信度：<b>{confidence*100:.1f}%</b></div>"

    # 概率分布
    pos_prob = float(proba[class_names.index("正面")]) if "正面" in class_names else 0
    neg_prob = float(proba[class_names.index("负面")]) if "负面" in class_names else 0
    prob_html = f"""
    <div style='margin-top:16px;'>
      <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
        <span>😊 正面</span><span><b>{pos_prob*100:.1f}%</b></span>
      </div>
      <div style='background:#e0e0e0;border-radius:8px;height:20px;overflow:hidden;'>
        <div style='background:#4caf50;height:100%;width:{pos_prob*100}%;border-radius:8px;transition:width 0.5s;'></div>
      </div>
      <div style='display:flex;justify-content:space-between;margin:12px 0 4px;'>
        <span>😞 负面</span><span><b>{neg_prob*100:.1f}%</b></span>
      </div>
      <div style='background:#e0e0e0;border-radius:8px;height:20px;overflow:hidden;'>
        <div style='background:#f44336;height:100%;width:{neg_prob*100}%;border-radius:8px;transition:width 0.5s;'></div>
      </div>
    </div>
    """

    # 分词结果
    tokens_html = f"<div style='margin-top:16px;color:#666;font-size:13px;'><b>分词结果：</b>{cut}</div>"

    return label_html + conf_html, prob_html, tokens_html, f"清洗后：{cleaned}"


# 示例评论
EXAMPLES = [
    "这个商品质量很好，发货速度快，非常满意，下次还会买",
    "东西太差了，用了两天就坏了，客服态度也不好，差评",
    "物流挺快的，包装也还行，东西一般般吧，没有想象中好",
    "性价比很高，推荐购买，五星好评",
    "完全不值这个价，质量差，退货了",
    "用了一段时间来评价，效果不错，值得入手",
]

with gr.Blocks(title="电商评论情感分析", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛒 电商评论情感分析系统\n基于TF-IDF特征 + 逻辑回归模型，对中文电商评论进行正面/负面二分类预测。验证集准确率89.1%，5折交叉验证87.3%。")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(label="输入评论", lines=4, placeholder="请输入一条中文电商评论...")
            analyze_btn = gr.Button("🔍 分析情感", variant="primary")
            gr.Markdown("**快速示例（点击填入）：**")
            example_buttons = gr.Examples(
                examples=EXAMPLES,
                inputs=input_text,
                label=None,
            )

        with gr.Column(scale=1):
            label_out = gr.HTML(label="预测结果")
            prob_out = gr.HTML(label="概率分布")
            tokens_out = gr.HTML(label="文本处理")
            cleaned_out = gr.Textbox(label="清洗后文本", interactive=False)

    analyze_btn.click(
        fn=analyze,
        inputs=input_text,
        outputs=[label_out, prob_out, tokens_out, cleaned_out],
    )

if __name__ == "__main__":
    demo.launch()
