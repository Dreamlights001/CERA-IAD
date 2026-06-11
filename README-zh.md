# CERA-IAD

[English](README.md) | [中文](README-zh.md)

**CERA-IAD: Conformal Evidence-Rarity Agent for Zero-shot Industrial Anomaly Detection**

CERA-IAD 是一个面向零样本工业异常检测的研究代码库。方法将语义异常候选、正常样本记忆检索、conformal calibration、mask 级定位和可选的多模态证据反思整合到统一评测流程中。

## 方法亮点

- 基于 evidence 的异常评分，融合 semantic、rarity、disagreement 和 uncertainty 信号。
- 使用 normal memory retrieval 估计跨类别视觉稀有性。
- 使用 conformal calibration 支持阈值校准和 review band 分析。
- 使用可提示分割模型进行 mask refinement。
- 提供 A0-A6 消融协议，便于逐项分析方法组件。

## 环境安装

```bash
cd /workspace/IAD/code/CERA-IAD

conda create -y -n cera python=3.10
conda activate cera
bash scripts/setup_env.sh
```

依赖统一维护在 `requirements.txt` 中。`scripts/setup_env.sh` 会根据目标机器的 CUDA 配置先安装 PyTorch，再安装项目依赖。

## 预训练权重

预训练权重按需下载，下载根目录由 `configs/cera_iad_config.yaml` 中的 `paths.weights_root` 控制。

查看权重清单和本地状态：

```bash
bash weights/download_weights.sh --list
```

预览下载计划，不访问网络：

```bash
bash weights/download_weights.sh --dry-run --basic
```

| Model | Purpose | Required for | Group | Source |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14 | 语义图像/patch 编码器 | A0-A6 | `basic` | `openai/clip-vit-large-patch14` |
| SAM ViT-H | box-prompt mask refinement | A4-A6 | `basic` | `facebook/sam-vit-huge` |
| DINOv2 Large | language-free encoder 消融 | encoder swap | `optional` | `facebook/dinov2-large` |
| SAM2.1 Hiera Large | mask refiner 消融 | mask swap | `optional` | `facebook/sam2.1-hiera-large` |
| Qwen2.5-VL-7B-Instruct | evidence reflection | A5-A6 | `mllm` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| InternVL2.5-8B | reasoner 消融 | reasoner swap | `mllm` | `OpenGVLab/InternVL2_5-8B` |
| AnomalyVFM checkpoints | encoder 消融 | manual swap | `manual` | 按上游 checkpoint 说明配置 |

按组下载：

```bash
bash weights/download_weights.sh --basic
bash weights/download_weights.sh --optional
bash weights/download_weights.sh --mllm
```

每个下载完成的模型目录都会包含 `.cera_weight.json`，记录来源仓库、下载时间、用途标签和本地文件数量。

## 数据准备

数据集路径在 `configs/cera_iad_config.yaml` 中配置。默认根目录为 `../../data`：

```text
data/
  MVTec-AD/
  MVTec-AD-2/
  VisA/
  Real-IAD/
  MMAD/
  M3-AD/
```

## 运行实验

单个消融 dry-run：

```bash
python scripts/run_cera.py \
  --config configs/cera_iad_config.yaml \
  --ablation A0_clip_only \
  --dry-run
```

生成全部消融计划：

```bash
bash scripts/run_ablation_matrix.sh
```

A0-A6 协议定义在 `configs/ablation_matrix.yaml`：

| ID | Description |
| --- | --- |
| A0 | CLIP semantic prior |
| A1 | anomaly-aware candidate generation |
| A2 | normal-memory rarity evidence |
| A3 | conformal calibration |
| A4 | SAM-based mask refinement |
| A5 | MLLM evidence reflection |
| A6 | full CERA-IAD |

## 结果

完整评测完成后在此更新 benchmark 结果。

| Dataset | Image AUROC | Pixel AUROC | AU-PRO | ECE | AURC |
| --- | --- | --- | --- | --- | --- |
| MVTec AD 2 | TBD | TBD | TBD | TBD | TBD |
| Real-IAD | TBD | TBD | TBD | TBD | TBD |
| MMAD | TBD | TBD | TBD | TBD | TBD |
| M3-AD | TBD | TBD | TBD | TBD | TBD |

## 目录结构

- `cera_iad/`：核心方法包。
- `configs/`：实验配置和消融矩阵。
- `scripts/`：环境安装与实验入口。
- `weights/`：预训练权重清单和下载工具。
- `tests/`：静态检查和 dry-run 检查。

## 致谢

CERA-IAD 使用公开基础模型，并可与外部异常检测 baseline 进行对比或可选适配。使用对应模型、checkpoint 或代码时，请同时引用其上游论文和仓库。
