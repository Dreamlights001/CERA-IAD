# CERA-IAD 云端实验工程

[English](README.md) | [中文](README-zh.md)

**CERA-IAD: Conformal Evidence-Rarity Agent for Zero-shot Industrial Anomaly Detection**

本目录现在是云服务器实验工程骨架。本地 Windows 只用于代码整理、配置检查和文档维护；正式特征提取、候选区域生成、SAM/MLLM 推理和消融实验都放到 Ubuntu + CUDA 云服务器运行。

## 目录结构

- `cera_iad/`：CERA-IAD Python package。核心链路仍是 `RegionProposal -> NormalMemory -> CeraIAD -> ImageEvidence`。
- `cera_iad/modules/`：模块注册表和云端 dry-run 计划生成器。
- `configs/cera_iad_cloud.yaml`：正式云端默认配置。
- `configs/ablation_matrix.yaml`：A0-A6 消融矩阵。
- `weights/`：统一预训练权重 manifest 和 Hugging Face 下载脚本，不提交大权重。
- `scripts/cloud_prepare.sh`：Ubuntu 环境准备脚本。
- `scripts/run_cera_cloud.py`：云端入口。当前支持安全 dry-run 和 adapter 计划输出。
- `scripts/run_ablation_matrix.sh`：批量生成/运行消融计划的入口。
- `baselines/`：已克隆并清理 Git 信息的顶会代码，用于 adapter 复用。

## 云服务器准备

```bash
cd /workspace/IAD/code/CERA-IAD
export IAD_ROOT=/workspace/IAD
export IAD_DATA=/data/iad
export IAD_OUTPUTS=/workspace/IAD/outputs/cera_cloud
export IAD_WEIGHTS=/workspace/IAD/code/CERA-IAD/weights/cache

bash scripts/cloud_prepare.sh
```

依赖文件：

- `requirements.txt`：dry-run 和 feature-level 工具所需的最小依赖。
- `requirements-cloud.txt`：云端实验依赖，不包含 PyTorch CUDA wheel。
- `environment.yml`：conda 环境封装，会安装 `requirements-cloud.txt`。

下载基础权重：

```bash
bash weights/download_weights.sh --basic
```

下载 MLLM 反思消融权重：

```bash
bash weights/download_weights.sh --mllm
```

可选替换模块权重：

```bash
bash weights/download_weights.sh --optional
```

## 模块替换方式

默认模块在 `configs/cera_iad_cloud.yaml`：

```yaml
modules:
  encoder: clip_openai
  candidate_generator: aa_clip
  memory_backend: exact_knn
  mask_refiner: sam_box
  reasoner: no_reasoner
  calibrator: conformal_global
  fusion_scorer: linear_fusion
```

可替换项由 `cera_iad/modules/registry.py` 注册：

- `encoder`：`clip_openai`、`dinov2_large`、`anomaly_vfm`
- `candidate_generator`：`clip_only`、`aa_clip`、`anomaly_clip`、`mrad`
- `memory_backend`：`exact_knn`、`faiss_knn`
- `mask_refiner`：`no_mask`、`sam_box`、`sam2_box`
- `reasoner`：`no_reasoner`、`qwen25_vl`、`internvl25`
- `calibrator`：`fixed_threshold`、`conformal_global`
- `fusion_scorer`：`linear_fusion`

设计原则：底层模型和顶会方法优先复用 `baselines/` 或 Hugging Face/package 接口，CERA-IAD 只写 adapter，把输出统一转换为 `RegionProposal`、`ImageEvidence` 和 metrics JSON。

## 消融实验

消融定义在 `configs/ablation_matrix.yaml`。

单个消融 dry-run：

```bash
python scripts/run_cera_cloud.py \
  --config configs/cera_iad_cloud.yaml \
  --ablation A0_clip_only \
  --dry-run
```

批量生成消融计划：

```bash
bash scripts/run_ablation_matrix.sh
```

消融项：

- `A0_clip_only`：仅 CLIP 语义先验。
- `A1_aa_clip_candidate`：替换为 AA-CLIP 候选区域。
- `A2_memory_rarity`：加入正常记忆检索和 rarity evidence。
- `A3_conformal_gate`：加入 conformal gate 和 review band。
- `A4_sam_mask`：加入 SAM box-prompt mask geometry evidence。
- `A5_mllm_reflection`：加入 Qwen2.5-VL evidence reflection。
- `A6_full_cera_iad`：完整 CERA-IAD。

更换中间模块时，优先改 `ablation_matrix.yaml` 中对应消融项的 `modules`，不要在脚本里硬编码。

## 权重和数据

权重统一由 `weights/weights_manifest.yaml` 管理，下载脚本尽量使用 Hugging Face：

- CLIP：`openai/clip-vit-large-patch14`
- DINOv2：`facebook/dinov2-large`
- SAM：`facebook/sam-vit-huge`
- SAM2.1：`facebook/sam2.1-hiera-large`
- Qwen2.5-VL：`Qwen/Qwen2.5-VL-7B-Instruct`
- InternVL2.5：`OpenGVLab/InternVL2_5-8B`

数据集不由脚本下载。云端按 `IAD_DATA` 放置：

```text
/data/iad/
  MVTec-AD/
  MVTec-AD-2/
  VisA/
  Real-IAD/
  MMAD/
  M3-AD/
```

## 输出

默认输出根目录：

```bash
${IAD_OUTPUTS:-outputs/cera_cloud}
```

建议保存：

- `features/normal_memory.npy`
- `calibration/normal_scores.json`
- `proposals/*.json`
- `evidence/*.json`
- `metrics/*.json`
- `plans/A0_clip_only.json` 到 `plans/A6_full_cera_iad.json`

正式论文筛选规则来自 `reviews/cera_iad_method_validation_zh.html`：如果模块只提升老 MVTec，不提升 MVTec AD 2 / Real-IAD，或显著伤害 calibration，就降级为 optional，不进入主方法。
