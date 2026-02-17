# MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the official implementation of:

**MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation**

---

## 📋 Overview

MapAgent is an agentic refinement framework built on top of a frozen BEV (Bird's Eye View) vectorization backbone. It introduces a bounded **Judge–Planner–Worker** loop to verify mapping specifications and perform deterministic, tool-grounded edits for lane-level map refinement in autonomous driving systems.

### Key Features

- 🎯 **Vision-Language Judge Agent** for automated quality assessment of lane predictions
- 🔄 **Bounded Refinement Loop** with deterministic worker tools
- 🚗 **Industrial-Grade** design for city-scale deployment
- 🧠 **GRPO-based** reinforcement learning for model optimization
- 📊 **Comprehensive** error taxonomy and structured reasoning

---

## 🗂️ Repository Structure

```
./
├── data_example/          # Training data and examples
│   ├── dataset/
│   │   ├── map_agent_dataset.jsonl
│   │   └── images/        # Lane prediction visualization
│   └── prompt/
│       └── train_example.txt
├── EasyR1/                # GRPO reinforcement learning
│   ├── generate_grpo_data.py
│   ├── grpo_config.yaml
│   ├── grpo.sh
│   ├── format_prompt/
│   │   └── map.jinja
│   └── reward/
│       └── map_reward.py
└── LlamaFactory/          # Model training configurations
    ├── inference/
    │   └── infer_qwen_lora_sft.yaml
    ├── merge/
    │   └── merge_qwen_lora_sft.yaml
    └── train/
        └── train_qwen_lora_sft.yaml
```

---

## 📦 Components

### 1. Data Example (`data_example/`)

Contains training datasets and prompt examples for the vision-language judge agent.

- **Dataset Structure**:
  - `map_agent_dataset.jsonl`: Annotated lane prediction examples with error labels
  - `images/`: Visualization of predicted lane lines overlaid on road images
  - `prompt/train_example.txt`: Complete prompt template with reasoning examples

- **Error Taxonomy**:
  - `extra_lane_line`: Predicted line on non-existent markings
  - `category_error`: Incorrect lane category classification
  - `geometry_error`: Local geometric defects (kinks, spikes)
  - `structure_error`: Major structural failures (wrong direction, large deviations)
  - `no_error`: Correct prediction

- **Ground Truth Categories**:
  ```
  1: Long dashed line      6: Short dashed line
  2: Double solid line     7: Double dashed line
  3: Single solid line     9: Virtual line
  4: Dashed-solid line    10: Stop line
  5: Solid-dashed line
  ```

### 2. GRPO Training (`EasyR1/`)

Group Relative Policy Optimization (GRPO) for reinforcement learning-based model refinement.

- **Core Scripts**:
  - `generate_grpo_data.py`: Generate training pairs for GRPO
  - `grpo.sh`: Training pipeline execution script
  - `grpo_config.yaml`: Hyperparameters and training configuration

- **Components**:
  - `format_prompt/map.jinja`: Jinja2 template for prompt formatting
  - `reward/map_reward.py`: Reward function for RL training

- **Purpose**: Fine-tune the judge agent to improve error detection accuracy through reinforcement learning.

### 3. Model Training (`LlamaFactory/`)

Configuration files for supervised fine-tuning (SFT) and model deployment using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

- **Training** (`train/`):
  - `train_qwen_lora_sft.yaml`: LoRA-based SFT configuration for Qwen models

- **Inference** (`inference/`):
  - `infer_qwen_lora_sft.yaml`: Inference configuration with LoRA adapters

- **Merge** (`merge/`):
  - `merge_qwen_lora_sft.yaml`: Configuration for merging LoRA weights into base model

- **Supported Models**: Qwen-VL, Qwen2-VL with LoRA parameter-efficient fine-tuning

---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/eadst/KDD-2026-MapAgent.git

```

### Training the Judge Agent

#### Option 1: Supervised Fine-Tuning (SFT)

```bash
# Using LlamaFactory
cd LlamaFactory/train
llamafactory-cli train train_qwen_lora_sft.yaml
```

#### Option 2: GRPO Reinforcement Learning

```bash
# Generate GRPO training data
cd EasyR1
python generate_grpo_data.py

# Run GRPO training
bash grpo.sh
```

### Inference

```bash
# Inference with trained model
cd LlamaFactory/inference
llamafactory-cli chat infer_qwen_lora_sft.yaml
```

### Model Merging

```bash
# Merge LoRA weights into base model
cd LlamaFactory/merge
llamafactory-cli export merge_qwen_lora_sft.yaml
```

---

## 📊 Dataset Format

The training data follows a structured format:

```json
{
  "image": "path/to/image.png",
  "predicted_category": "double_solid_line",
  "bounding_box": [0.0, 1473.0, 1536.0, 11.0],
  "reasoning": "<think>Analysis process...</think>",
  "error_type": "no_error",
  "gt_category_id": 2
}
```

---

## 🎯 Usage Example

```python
# Example: Evaluate a lane prediction
from mapagent import JudgeAgent

# Initialize judge agent
judge = JudgeAgent(model_path="path/to/model")

# Evaluate prediction
result = judge.evaluate(
    image_path="data_example/dataset/images/557_5950_pred.png",
    predicted_category="double_solid_line",
    bounding_box=[0.0, 1473.0, 1536.0, 11.0]
)

print(f"Error Type: {result['error_type']}")
print(f"Reasoning: {result['reasoning']}")
print(f"GT Category: {result['gt_category_id']}")
```

---

## 📈 Evaluation Protocol

The judge agent follows a strict priority-based evaluation:

1. **Priority Order**: `extra_lane_line` → `category_error` → `geometry_error` → `structure_error` → `no_error`
2. **Short-Circuit Logic**: Stop evaluation once an error is confirmed
3. **Visual Decoding Strategy**: Distinguish between prediction mask (hypothesis) and road texture (fact)
4. **Structured Reasoning**: Output reasoning process within `<think>` tags

---

## 🔧 Configuration

### GRPO Training Parameters

```yaml
# grpo_config.yaml
model_name: Qwen2-VL-7B-Instruct
learning_rate: 5.0e-6
num_train_epochs: 2
per_device_train_batch_size: 2
reward_model: map_reward
```

### SFT Training Parameters

```yaml
# train_qwen_lora_sft.yaml
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
learning_rate: 5.0e-5
```

---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{mapagent2025,
  title={MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation},
  author={Author Names},
  journal={Conference/Journal Name},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training infrastructure
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for vision-language foundation models
- EasyR1 framework for GRPO implementation

---

## 📧 Contact

For questions and discussions, please open an issue:
- **Project Page**: [https://github.com/eadst/KDD-2026-MapAgent](https://github.com/eadst/KDD-2026-MapAgent)

---

## 🚧 Status

**Current Version**: v0.1.0 (Initial Release)

- ✅ Data examples and prompt templates
- ✅ GRPO training pipeline
- ✅ LlamaFactory configurations
- 🚧 Full pipeline integration (coming soon)
- 🚧 Pre-trained model weights (coming soon)
- 🚧 Evaluation scripts (coming soon)

---

## 🗺️ Roadmap

- [ ] Release pre-trained judge agent weights
- [ ] Add full pipeline integration code
- [ ] Provide evaluation benchmarks
- [ ] Add worker tools implementation
- [ ] Release technical report
- [ ] Add interactive demo

---

**Last Updated**: February 17, 2026
