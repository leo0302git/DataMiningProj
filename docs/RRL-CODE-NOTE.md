# RRL 源码阅读笔记

本文记录本项目固定上游快照 `f8d0886` 的源码结构、调用关系和实现细节，用于在后续适配 DIA、修复兼容性或修改模型结构时判断改动边界。算法原理、论文公式与指标解释统一放在 `RRL-CONCEPT-NOTE.md`，本文不重复推导。当前 RRL 位于 [`models/rrl`](../models/rrl)，是 DM 仓库中的普通代码目录，不是 submodule 或 subtree。

## 1. 源码分层与执行主线

| 文件                                                    | 主要职责                                                           | 是否属于 RRL 核心算法                    |
| ------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------- |
| [`args.py`](../models/rrl/args.py)                     | 解析命令行参数，创建实验目录和输出路径                             | 否，属于实验配置                         |
| [`experiment.py`](../models/rrl/experiment.py)         | 组织数据加载、训练、模型重载、测试、规则导出和边数统计             | 大部分不是；数据划分会影响实验正确性     |
| [`rrl/utils.py`](../models/rrl/rrl/utils.py)           | 读取`.data/.info`，完成输入编码、缺失值填充和标准化              | 否，属于模型输入适配                     |
| [`rrl/models.py`](../models/rrl/rrl/models.py)         | `Net` 组装网络，`RRL` 管理训练、评价、保存和规则提取           | `Net` 的层组织和训练路径会触及模型行为 |
| [`rrl/components.py`](../models/rrl/rrl/components.py) | 实现二值化层、AND/OR 层、NLAF、Gradient Grafting、线性层和规则还原 | 是，核心实现集中于此                     |

程序入口位于 `experiment.py` 底部。一次运行先从 `args.py` 得到 `rrl_args`，调用 `train_main()` 启动训练，再调用 `test_model()` 从 `model.pth` 重新加载模型、测试、打印规则并计算 `log(#edges)`。当前调用链可以压缩为：

```text
args.py
  → experiment.train_main()
  → experiment.train_model()
      → get_data_loader()
      → RRL(...)
      → RRL.train_model()
  → experiment.test_model()
      → load_model()
      → RRL.test()
      → RRL.rule_print()
      → log(#edges)
```

`train_model()` 从编码器读取 `discrete_flen`、`continuous_flen` 和类别数，再把 `-s` 参数转换成 `dim_list`。例如 `-s 5@64` 且有两个类别时，结构为 `[(discrete_flen, continuous_flen), 5, 64, 2]`：第一个元组描述输入布局，5 是每个连续特征的候选阈值数，64 是 `UnionLayer` 中合取节点和析取节点各自的数量，2 是线性层输出的 logits 数。`RRL` 是训练控制器而不是 `nn.Module`；真正继承 `nn.Module` 的是其中的 `Net`。`Net.forward()` 依次执行二值化层、逻辑层和线性层，`RRL` 则负责损失、优化器、评价、保存、死节点检测和规则导出。

## 2. `DBEncoder` 的位置与职责

`DBEncoder` 容易因名称产生误解：它不是将连续特征变成 `x>t` 和 `x≤t` 的模型二值化层。它只完成进入网络前的表格预处理；真正生成阈值条件的是 `components.py` 中的 `BinarizeLayer`。两者的边界是：

```text
原始表格
  → DBEncoder：离散特征 One-Hot；连续特征填补和标准化
  → [离散 0/1 列 | 连续标准化列]
  → BinarizeLayer：连续列变成 x>t 与 x≤t 条件
```

### 2.1 `.data`、`.info` 与 `read_csv()`

`.data` 是无表头数据，每行一个样本；`.info` 的每一行给出对应列的名称和 `discrete/continuous` 类型，最后一行 `LABEL_POS` 指定标签位置。`read_info()` 返回特征描述和标签索引；`read_csv()` 据此给 DataFrame 命名，得到 `X_df`、`y_df`、去掉标签行后的 `f_df` 和 `label_pos`。特征类型并非由 RRL 自动推断，而是在制备 `.info` 时决定。

### 2.2 初始化的三个转换器

`DBEncoder(f_df, discrete=False, y_one_hot=True, drop='first')` 保存特征描述，并创建三个 sklearn 转换器：`label_enc` 对标签做 One-Hot，DIA 的两类因此对应两个位置；`feature_enc` 对离散输入做 One-Hot，默认删除每个离散特征的第一个基准类别；`SimpleImputer(strategy='mean')` 用训练数据均值填补连续特征缺失值。参数名 `discrete=False` 较易混淆：它不表示“没有离散特征”，而表示输出中除了 One-Hot 离散列外仍保留连续列；设为 `True` 时当前实现只输出离散部分。

### 2.3 `split_data()`

`split_data()` 按 `f_df` 中的类型标记选出离散列和连续列，并把连续列中的旧式缺失标记 `?` 替换为 `NaN`，再转换成浮点数。对 DIA，原生 0–1 描述符适合作为 `discrete`；一般整数计数仍宜作为 `continuous`，因为其数值顺序应交给阈值条件表达，而不是把 0、1、2、3 等每个数值当成互不相关的类别。

### 2.4 `fit()` 学习转换参数

`fit(X_df, y_df)` 先拆分两类输入，然后依次学习：标签类别及其输出名称；连续特征的均值填充值；离散特征出现过的类别及 One-Hot 列名。它还保存 `X_fname`、`y_fname`、`discrete_flen` 和 `continuous_flen`。后二者是编码器与网络的关键接口：输入矩阵始终按 `[X_discrete | X_continuous]` 排列，`BinarizeLayer` 依靠 `discrete_flen` 知道前多少列已经是逻辑值、后多少列仍需阈值化。

离散特征使用 `OneHotEncoder(drop='first')`。一个有 $k$ 个类别的特征保留 $k-1$ 列；对取值为 `{0,1}` 的二值特征通常只保留一列，取值 0 代表基准类别，取值 1 代表另一类别。标签编码器不删除基准类，因此 DIA 的 `y` 仍有两列。

### 2.5 `transform()` 生成网络输入

`transform()` 使用 `fit()` 学到的转换器，而不应重新估计参数。标签先转换成 One-Hot 数组；连续特征先按训练阶段学到的均值填补缺失值，再在 `normalized=True` 时标准化：

$$
x' = \frac{x-\mu}{s}
$$

当 `keep_stat=True` 时，当前实现把本次输入的均值和标准差保存到 `self.mean/self.std`。离散部分完成 One-Hot 后，与标准化连续部分按固定顺序拼接，最后返回 NumPy 数组 `X, y`。这些均值和标准差不仅用于输入转换；规则导出时还通过 $c_{raw}=cs+\mu$ 把标准化空间中的候选阈值还原成原始单位。

若 DIA 暂按 20 个二值输入和 176 个连续输入制备，DBEncoder 输出仍约为 196 维；使用 `BinarizeLayer(n=5)` 后才扩展为：

$$
D_{out}=20+2\times5\times176=1780
$$

其中乘以 2 是因为每个候选阈值同时显式生成 `x>t` 与 `x≤t` 两个条件。

### 2.6 当前调用中的问题

`experiment.get_data_loader()` 目前先在全部数据上执行 `db_enc.fit()` 和保存标准化统计量，再进行五折划分，因此留出折参与了缺失值填充值、One-Hot 类别和均值/标准差的估计，属于预处理泄漏。正确顺序应为“先划分索引，再只在训练折 `fit`，最后分别 `transform` 训练折、验证折和留出折”。当前实现还使用已被新版 NumPy/sklearn 删除的 `np.float` 和 `get_feature_names()`，没有处理零标准差，也没有允许留出折出现训练折未见的离散类别。这些属于兼容性和实验正确性修复，不改变 RRL 的核心数学思想。
