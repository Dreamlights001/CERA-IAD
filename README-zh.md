# CERA-IAD

[English](README.md) | [中文](README-zh.md)

**CERA-IAD: Conformal Evidence-Rarity Agent for Zero-shot Industrial Anomaly Detection**

CERA-IAD 是面向 Ubuntu 的零样本工业异常检测研究代码库，集成可复用的 VFM/VLM backbone、顶会异常检测基线、正常记忆检索、conformal 校准、mask refinement，以及可选的 MLLM evidence reflection。

## 目录结构

- `cera_iad/`：CERA-IAD Python package。核心链路仍是 `RegionProposal -> NormalMemory -> CeraIAD -> ImageEvidence`。
- `cera_iad/modules/`：模块注册表和实验计划生成器。
- `configs/cera_iad_config.yaml`：总实验配置。
- `configs/ablation_matrix.yaml`：A0-A6 消融矩阵。
- `weights/`：统一预训练权重 manifest 和 Hugging Face 下载脚本，不提交大权重。
- `scripts/setup_env.sh`：Ubuntu 环境准备脚本。
- `scripts/run_cera.py`：实验入口。当前支持 dry-run 和 adapter 计划输出。
- `scripts/run_ablation_matrix.sh`：批量生成/运行消融计划的入口。
- `baselines/`：已克隆并清理 Git 信息的顶会代码，用于 adapter 复用。

## 环境准备

```bash
cd /workspace/IAD/code/CERA-IAD

bash scripts/setup_env.sh
```

所有路径统一在 `configs/cera_iad_config.yaml` 中配置。依赖统一写在一个 `requirements.txt` 里。`setup_env.sh` 会先按 CUDA 版本安装 PyTorch，再安装 `requirements.txt`。

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

默认模块在 `configs/cera_iad_config.yaml`：

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
python scripts/run_cera.py \
  --config configs/cera_iad_config.yaml \
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

数据集不由脚本下载。默认数据根目录在 `configs/cera_iad_config.yaml` 中配置为 `../../data`：

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

默认输出根目录在 `configs/cera_iad_config.yaml` 中配置：

```bash
../../outputs/cera_experiments
```

建议保存：

- `features/normal_memory.npy`
- `calibration/normal_scores.json`
- `proposals/*.json`
- `evidence/*.json`
- `metrics/*.json`
- `plans/A0_clip_only.json` 到 `plans/A6_full_cera_iad.json`

正式论文筛选规则来自 `reviews/cera_iad_method_validation_zh.html`：如果模块只提升老 MVTec，不提升 MVTec AD 2 / Real-IAD，或显著伤害 calibration，就降级为 optional，不进入主方法。
