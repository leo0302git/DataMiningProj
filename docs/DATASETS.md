# 数据来源

当前使用的原始数据纳入 Git，也可通过 `download_data.py` 从下列 UCI 官方数据包重新下载并校验。

## Drug Induced Autoimmunity Prediction

- UCI 数据集 ID：1104
- 来源：https://archive.ics.uci.edu/dataset/1104/drug_induced_autoimmunity_prediction
- 必需原始文件：`DIA_trainingset_RDKit_descriptors.csv`、`DIA_testset_RDKit_descriptors.csv`
- `RDKit_ChemDes.xlsx` 保留为描述符说明。
- 两份 CSV 可合并后进行 K 折交叉验证；`SMILES` 是标识而非模型特征，`Label` 是二分类目标。

## Abalone

- UCI 数据集 ID：1
- 来源：https://archive.ics.uci.edu/dataset/1/abalone
- 必需原始文件：`abalone.data`、`abalone.names`
- 回归目标是 `Rings`，不是派生年龄 `Rings + 1.5`。
