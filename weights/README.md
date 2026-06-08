# CERA-IAD 权重目录

本目录只放权重清单和下载脚本，不提交大模型文件。正式下载请在 Ubuntu 云服务器上执行。

## 快速下载

```bash
cd /workspace/IAD/code/CERA-IAD
python -m pip install -U huggingface_hub
bash weights/download_weights.sh --basic
```

`--basic` 下载 CLIP 和 SAM，支持 A0-A4 及不启用 MLLM 的实验。

大模型反思消融需要额外下载：

```bash
bash weights/download_weights.sh --mllm
```

可选模块：

```bash
bash weights/download_weights.sh --optional
```

## 模型分组

- `basic`：`openai/clip-vit-large-patch14`、`facebook/sam-vit-huge`
- `optional`：`facebook/dinov2-large`、`facebook/sam2.1-hiera-large`
- `mllm`：`Qwen/Qwen2.5-VL-7B-Instruct`、`OpenGVLab/InternVL2_5-8B`

详细用途见 `weights_manifest.yaml`。

## 路径约定

默认下载到：

```bash
weights/cache
```

也可以用环境变量覆盖：

```bash
export IAD_WEIGHTS=/mnt/models/iad
bash weights/download_weights.sh --basic
```

数据集、闭源 API key 和各论文作者未公开的专用 checkpoint 不在本脚本内下载，请按对应 baseline README 单独配置。
