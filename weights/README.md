# CERA-IAD 权重目录

本目录用于管理 CERA-IAD 的预训练权重清单、下载入口和本地缓存状态。权重文件按需下载到 `configs/cera_iad_config.yaml` 中的 `paths.weights_root`，默认位置为：

```text
weights/cache
```

## 快速开始

列出所有权重和本地状态：

```bash
bash weights/download_weights.sh --list
```

预览 basic 组下载计划，不访问网络：

```bash
bash weights/download_weights.sh --dry-run --basic
```

下载 basic 组权重：

```bash
bash weights/download_weights.sh --basic
```

下载可选消融权重：

```bash
bash weights/download_weights.sh --optional
```

下载 MLLM reasoner 权重：

```bash
bash weights/download_weights.sh --mllm
```

## 权重分组

| Group | Models | 用途 |
| --- | --- | --- |
| `basic` | CLIP ViT-L/14, SAM ViT-H | 默认编码器和 A4-A6 的 mask refinement |
| `optional` | DINOv2 Large, SAM2.1 Hiera Large | encoder / mask refiner 替换消融 |
| `mllm` | Qwen2.5-VL-7B-Instruct, InternVL2.5-8B | evidence reflection 与 reasoner 替换消融 |
| `manual` | AnomalyVFM checkpoints | 上游 checkpoint 未统一纳入自动下载 |

下载完成后，每个模型目录会写入 `.cera_weight.json`，记录来源、分组、下载时间、用途和本地文件数。再次运行下载命令时，已有该元数据的模型会被识别为 `already available`；如需重新下载，使用：

```bash
bash weights/download_weights.sh --force --basic
```

## 实现说明

下载入口 `weights/download_weights.sh` 调用：

```bash
python -m cera_iad.tools.download_weights
```

实际下载使用 `huggingface_hub.snapshot_download`，不再依赖已废弃的 `huggingface-cli`。如遇到 gated model 或私有仓库，请先在服务器上执行 Hugging Face 登录：

```bash
hf auth login
```

完整模型清单见 `weights_manifest.yaml`。
