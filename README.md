# DM

2026 年秋季数据挖掘课程大作业：在两个表格数据集上比较可解释模型与基线模型。

## 项目范围

- DIA：使用 RDKit 数值描述符进行二分类；`SMILES` 仅作为标识，不参与建模。
- Abalone：以 `Rings` 为回归目标。
- 模型：RRL、线性模型、树集成模型与手写广义线性模型。

## 目录说明

- `data/`：本地 UCI 原始数据与处理后数据，不提交到 Git。
- `models/rrl/`：固定版本的上游 RRL 副本，来源见其中的 `UPSTREAM.md`。
- `models/linear/`、`models/tree_ensemble/`、`models/manual_glm/`：独立对比模型。
- `download_data.py`：下载并校验课程所需的官方原始数据。

## 初始化

```bash
uv sync
python download_data.py
```

下载脚本只获取 UCI 官方包并校验必需文件。数据、日志、模型权重和实验结果均不提交到 Git。
