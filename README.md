# DM

2026 年秋季数据挖掘课程大作业：在两个表格数据集上比较可解释模型与基线模型。

## 项目范围

- DIA：使用 RDKit 数值描述符进行二分类；`SMILES` 仅作为标识，不参与建模。
- Abalone：以 `Rings` 为回归目标。
- 模型：RRL、线性模型、树集成模型与手写广义线性模型。

## 目录说明

- `data/raw/`：纳入 Git 的 UCI 官方原始数据；体积合理的处理后数据也默认追踪。
- `analysis/`：可复现的数据分析与作图脚本。
- `results/eda/`：纳入 Git 的 EDA 统计表与精选图片。
- `models/rrl/`：固定版本的上游 RRL 副本，来源见其中的 `UPSTREAM.md`。
- `models/linear/`、`models/tree_ensemble/`、`models/manual_glm/`：独立对比模型。
- `download_data.py`：下载并校验课程所需的官方原始数据。

## 初始化

```bash
uv sync
python download_data.py
```

仓库已包含当前使用的 UCI 官方原始数据；下载脚本用于缺失时重新获取并校验。脚本、文档、体积合理的数据与实验结果默认纳入 Git；训练日志和模型权重仍忽略，未来出现的大文件再按具体文件处理。
