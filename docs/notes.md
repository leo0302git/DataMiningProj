对大作业任务的理解：

对于DIA任务：给的是两个表格的数据

| 数据              | DIA-positive | DIA-negative | 总数 |
| ----------------- | -----------: | -----------: | ---: |
| Training set      |          118 |          359 |  477 |
| External test set |           30 |           90 |  120 |
| 总计              |          148 |          449 |  597 |

可以理解每一行是一个药物分子；UCI 简介写约 195 个描述符，实际 CSV 中有 196 个数值描述符和一个 0/1 标签，任务是通过这些描述符进行分类，判断每个药物的 0/1 标签。

其他相术语：**SMILES**是化学式的一个完整表述，理论上包含了所有需要的信息。**RDKit**是一个确定性程序，通过 SMILES 表达式确定性地计算出一些方便数据分析的特征，比如分子量、结构特征等等；我们获得的数据就是 500+ 个药物分子乘以每个分子的 196 个 RDKit 数值描述符。

对于abalone任务：一句话，就是通过鲍鱼一些简单、易测量的数据来预测它的年龄（难测量）。给了一个4177行，9列的表格数据，每一行代表一只鲍鱼。9列中，有8列是鲍鱼的8个属性，最后一列称为 Rings，是我们要预测的目标（年龄 ≈ Rings + 1.5，所以预测Rings就是预测年龄）。

| 字段               | 类型       | 含义                                               |
| ------------------ | ---------- | -------------------------------------------------- |
| `Sex`            | 类别型     | 性别：`M`雄性、`F`雌性、`I`幼体              |
| `Length`         | 连续数值   | 贝壳最长尺寸                                       |
| `Diameter`       | 连续数值   | 垂直于长度方向的贝壳直径                           |
| `Height`         | 连续数值   | 包含肉体时的贝壳高度                               |
| `Whole weight`   | 连续数值   | 整只鲍鱼重量                                       |
| `Shucked weight` | 连续数值   | 去壳后肉的重量                                     |
| `Viscera weight` | 连续数值   | 放血后内脏重量                                     |
| `Shell weight`   | 连续数值   | 干燥后贝壳重量                                     |
| `Rings`          | 整数型目标 | 贝壳生长环数量，用于预测年龄，是本次任务的预测目标 |

Lecture02笔记

现实世界的数据是很脏的，可能包含不完整的数据。或者前后不一致的数据，以及重复的数据，还有错误和 Outlier。

数据预处理需要做的内容：

- Fill in missing values, smooth noisy data, identify or remove outliers, and resolve inconsistencies（需检查）
- 需要处理跨表的一致性（本大作业似乎不需要）
- Normalization and aggregation
- reduction：减少数据体量，却能产出几乎相同的分析结果。
- Part of data reduction but with particular importance, especially for numerical data
- numericalization比如独热编码，或者文本、视频的嵌入编码

Data summarization：

- central tendency, variation and spread
- Data dispersion characteristics：Median, max, min, quantiles, outliers, variance, etc.
- Transform numerical dimensions to sorted intervals：Boxplot or quantile analysis on sorted intervals
