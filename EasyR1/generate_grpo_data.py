import io
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ================== 路径配置 ==================

DATA_ROOT = Path("data")

TRAIN_JSONL = "data/mapagent_train.jsonl"
VAL_JSONL   = "data/mapagent_val.jsonl"

OUTPUT_DIR = Path(
    "data"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PARQUET = OUTPUT_DIR / "train.parquet"
VAL_PARQUET   = OUTPUT_DIR / "validation.parquet"

TRAIN_SAMPLE_SIZE = 6000
RANDOM_SEED = 42


# ================== 工具函数 ==================

def image_to_bytes(image_path: Path) -> bytes:
    """
    ⚠️ 关键：只返回 bytes
    这是 HF datasets + Ray + EasyR1 唯一稳定的 image 格式
    """
    with Image.open(image_path).convert("RGB") as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def parse_sharegpt_item(item):
    system_msg = ""
    user_msg = ""
    assistant_msg = ""

    for c in item["conversations"]:
        if c["from"] == "system":
            system_msg = c["value"]
        elif c["from"] == "user":
            user_msg = c["value"]
        elif c["from"] == "assistant":
            assistant_msg = c["value"]

    assert "<image>" in user_msg
    assert assistant_msg.strip() != ""

    problem = system_msg.strip() + "\n" + user_msg.strip()
    solution = assistant_msg.strip()

    return problem, solution


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ================== 主流程 ==================

def build_parquet(jsonl_path: Path, output_path: Path, sample_size=None):
    print(f"\nLoading {jsonl_path}")
    data = load_jsonl(jsonl_path)

    if sample_size is not None:
        random.seed(RANDOM_SEED)
        data = random.sample(data, sample_size)
        print(f"Sampled {sample_size} items")

    records = []

    for item in tqdm(data):
        assert len(item["images"]) == 1
        image_rel_path = item["images"][0]
        image_abs_path = DATA_ROOT / image_rel_path
        assert image_abs_path.exists()

        image_bytes = image_to_bytes(image_abs_path)
        problem, solution = parse_sharegpt_item(item)

        records.append({
            "image": image_bytes,   
            "problem": problem,
            "solution": solution
        })

    df = pd.DataFrame(records)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Saved {len(df)} samples -> {output_path}")


# ================== 执行 ==================

if __name__ == "__main__":
    build_parquet(
        TRAIN_JSONL,
        TRAIN_PARQUET,
        sample_size=TRAIN_SAMPLE_SIZE
    )

    build_parquet(
        VAL_JSONL,
        VAL_PARQUET,
        sample_size=None
    )
