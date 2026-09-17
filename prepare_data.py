# -*- coding: utf-8 -*-
"""
prepare_data.py
数据集准备脚本：将 ChnSentiCorp 中文情感分析 arrow 数据集转换为 CSV，并按 80/20 划分训练集/验证集

数据集：ChnSentiCorp（公开中文在线评论情感分析数据集，含酒店、电子产品、书籍等评论，二分类：0=负面，1=正面）
原始划分：train 9600 / validation 1200 / test 1200
本脚本合并后重新按 80/20 分层随机划分，保证正负样本比例一致。

用法：
    python prepare_data.py
"""
import random
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

RAW_DIR = Path("data_raw")
DATA_DIR = Path("data")
TRAIN_RATIO = 0.8
SEED = 42


def read_arrow(path: Path) -> pd.DataFrame:
    with pa.memory_map(str(path), "r") as f:
        table = ipc.open_stream(f).read_all()
    return table.to_pandas()


def main():
    random.seed(SEED)

    # 读取三个划分并合并
    dfs = []
    for split in ("train", "validation", "test"):
        fp = RAW_DIR / f"chnsenticorp-{split}.arrow"
        df = read_arrow(fp)
        dfs.append(df)
        print(f"  {split:10s}: {len(df)} 条")
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"\n合并后总计: {len(all_df)} 条")
    print(f"标签分布:\n{all_df['label'].value_counts()}")

    # 按标签分层随机划分
    pos = all_df[all_df["label"] == 1].sample(frac=1, random_state=SEED).reset_index(drop=True)
    neg = all_df[all_df["label"] == 0].sample(frac=1, random_state=SEED).reset_index(drop=True)

    n_pos_train = int(len(pos) * TRAIN_RATIO)
    n_neg_train = int(len(neg) * TRAIN_RATIO)

    train_df = pd.concat([pos[:n_pos_train], neg[:n_neg_train]], ignore_index=True)
    val_df = pd.concat([pos[n_pos_train:], neg[n_neg_train:]], ignore_index=True)

    # 打乱顺序
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(DATA_DIR / "val.csv", index=False, encoding="utf-8")

    print(f"\n划分完成：训练集 {len(train_df)} 条（正面{(train_df['label']==1).sum()} / 负面{(train_df['label']==0).sum()}），"
          f"验证集 {len(val_df)} 条（正面{(val_df['label']==1).sum()} / 负面{(val_df['label']==0).sum()}）")
    print(f"数据目录：{DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
