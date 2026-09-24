# RRL 学习笔记

本文记录对 Rule-based Representation Learner（RRL）的逐步学习，只纳入已经讲解并确认的内容。排版以连续论述为主，后续内容在完成讲解后继续补充。

## 1. 模型概览

RRL 是一种内生可解释模型：它先把原始特征转换成 0/1 条件，再用合取（AND）与析取（OR）构造非模糊规则，最后由线性层根据规则是否成立计算类别得分。其基本数据流为“原始特征 → 二值条件 → 逻辑规则 → 类别得分”。与事后解释方法不同，RRL 导出的规则就是模型实际执行的决策结构。

例如，DIA 中可以把两个条件写成：

$$A=(\texttt{MolWt}>300),\qquad B=(\texttt{fr\_aniline}>0)$$

合取规则 $R_1=A\land B$ 表示两个条件必须同时成立，析取规则 $R_2=C\lor D$ 表示至少一个条件成立。多层逻辑结构还能继续组合已有规则，例如 $R_3=R_1\lor R_2$。这使 RRL 能表达特征交互，适合处理 DIA 中“单个描述符作用较弱、多个条件组合后才可能具有区分力”的情况。本地 [`Net`](../models/rrl/rrl/models.py) 依次构造 `BinarizeLayer`、一个或多个 `UnionLayer` 和 `LRLayer`。

### 1.1 线性输出层

设最后一个逻辑层产生 $m$ 条二值规则 $R_j(x)\in\{0,1\}$。DIA 是二分类任务，因此模型分别计算阴性和阳性得分：

$$z_{\mathrm{negative}}=b_0+\sum_{j=1}^{m}w_{0j}R_j,\qquad z_{\mathrm{positive}}=b_1+\sum_{j=1}^{m}w_{1j}R_j$$

规则成立时 $R_j=1$，其权重会加入相应类别的得分；规则不成立时 $R_j=0$，贡献为零。偏置 $b_0,b_1$ 表示没有规则提供额外证据时对两个类别的基础倾向。假设三条规则的输出为 $(R_1,R_2,R_3)=(1,0,1)$，阴性权重为 $(-0.2,0.6,0.1)$、阳性权重为 $(0.8,-0.3,1.1)$，偏置分别为 $0.2$ 和 $-0.4$，则：

$$z_{\mathrm{negative}}=0.2-0.2+0+0.1=0.1,\qquad z_{\mathrm{positive}}=-0.4+0.8+0+1.1=1.5$$

因为 $z_{\mathrm{positive}}>z_{\mathrm{negative}}$，模型预测为阳性。判断一条规则更支持哪个类别时，不能只看某个权重是否为正，而应比较两个类别权重之差：

$$\Delta w_j=w_{\mathrm{positive},j}-w_{\mathrm{negative},j}$$

$\Delta w_j>0$ 表示规则成立时相对更支持阳性，$\Delta w_j<0$ 表示相对更支持阴性。本地 [`LRLayer`](../models/rrl/rrl/components.py) 使用 `nn.Linear(input_dim, output_dim)` 实现这一计算：`input_dim` 是规则数，DIA 的 `output_dim` 为 2，权重矩阵保存每条规则对两个类别的贡献，偏置向量保存两个类别的基础得分。

## 2. 输入编码与二值化

RRL 的逻辑层只能处理 0 和 1，因此数据先由 [`DBEncoder`](../models/rrl/rrl/utils.py) 根据 `.info` 文件中的 `discrete` 或 `continuous` 标记分流，再由 [`BinarizeLayer`](../models/rrl/rrl/components.py) 生成逻辑条件。特征类型不是 RRL 自动推断的，而是数据制备阶段明确指定的。针对 DIA，当前合理方案是删除 17 个常量特征，把 20 个真正的 0–1 特征作为离散输入，把其余 76 个整数计数和 83 个浮点特征作为连续输入；计数特征虽然取整数，但数值有顺序，使用“计数是否超过阈值”通常比把每个整数当作独立类别更自然。

### 2.1 离散特征与基准类别

`DBEncoder` 默认使用 `OneHotEncoder(drop="first")`。准确地说，一个具有 $k$ 个类别的离散特征，完整 One-Hot 原本产生 $k$ 列，删除第一个基准类别后产生 $k-1$ 列；不是整个数据集的列数简单减一。完整编码满足：

$$d_1+d_2+\cdots+d_k=1$$

当模型含偏置项时，完整的 $k$ 列存在确定的线性依赖，因此线性模型常删除一列。例如三类别特征保留后两个指示变量时，$(0,0)$ 表示基准类别，$(1,0)$ 和 $(0,1)$ 表示另外两个类别。对 20 个二值特征而言，每个特征由两列压缩为一列，最终仍是 20 列，而不是 19 列。

删除基准列并不是逻辑模型的硬性要求。它节省维度，却使基准类别表现为全 0；如果逻辑层既没有对应的正向节点也不允许 NOT，就不能直接写出“属于基准类别”的规则。RRL 可以通过保留完整 One-Hot、允许 NOT，或对二值特征保留一个比特并按需提供其反值来解决。因而 `drop="first"` 是当前编码器沿用的工程选择，而不是 RRL 必须遵守的数学原则。

### 2.2 连续特征的标准化与候选阈值

连续特征先按训练数据的均值和标准差进行标准化：

$$x' = \frac{x-\mu_{\mathrm{train}}}{s_{\mathrm{train}}}$$

对于连续特征 $x_j$，二值化层准备 $n$ 个候选切分点 $c_{j1},\ldots,c_{jn}$，并为每个切分点生成两个条件：

$$b^{>}_{jk}=\mathbf 1[x_j>c_{jk}],\qquad b^{\le}_{jk}=\mathbf 1[x_j\le c_{jk}]=1-b^{>}_{jk}$$

例如 $x=1$、候选阈值为 $(-0.5,0.2,1.3)$ 时，大于条件输出 $(1,1,0)$，小于等于条件输出 $(0,0,1)$。多个阈值不仅允许模型选择更合适的切分位置，还能通过合取表达区间：

$$(x>c_1)\land(x\le c_2)\iff c_1<x\le c_2$$

当前源码中的候选阈值 `cl` 通过 `register_buffer` 保存，而不是优化器更新的可训练参数；默认情况下，它们在标准化空间中从标准正态分布随机生成。如果提供特征上下界，则在相应范围内随机生成。模型所谓的“自动离散化”在这份实现中主要是由后续逻辑层从候选阈值池中选择并组合条件，而不是持续移动阈值本身。导出规则时，[`get_bound_name`](../models/rrl/rrl/components.py) 会按 $c_{\mathrm{raw}}=c\,s+\mu$ 把标准化阈值还原到原始单位。

### 2.3 为什么显式保存正反条件

$x>c$ 和 $x\le c$ 在信息论上完全互补，因此同时保存确实使用了两倍比特数；目的不是增加信息，而是把正、反两个“逻辑文字”都变成可被后续 AND/OR 层直接选择的激活节点。如果只提供 $a=\mathbf 1[x>c]$，那么 $x\le c$ 必须写成 $\neg a$；当下一层只会选择取值为 1 的输入且不额外实现 NOT 时，它无法直接把“$a=0$”作为条件。显式提供 $a$ 与 $1-a$ 后，规则 $(x\le c)\land(y>d)$ 可以直接连接两个取值为 1 的节点，规则导出也更直观。

这是一种“用空间换结构简单”的设计：代价是二值化层变宽，收益是第一逻辑层无需额外处理否定边。理论上可以只保留一个比特，再用 NOT 或带符号连接表达反条件，但这需要同步修改逻辑层和规则导出，不能只删除一半输入。

### 2.4 二值化层宽度

命令参数 `-s 5@64` 中的 5 表示每个连续特征使用 5 个候选阈值，64 表示后续 `UnionLayer` 分别包含 64 个合取节点和 64 个析取节点，并不是“5 层和 64 层”；该逻辑层拼接后实际输出 128 个节点。若 DIA 删除常量后有 20 个离散特征和 159 个连续特征，则二值化层输出宽度约为：

$$20+159\times5\times2=1610$$

其中乘以 2 正是因为每个阈值同时产生大于和小于等于条件。源码中的一般公式为：

$$D_{\mathrm{out}}=D_{\mathrm{discrete}}+2nD_{\mathrm{continuous}}$$

若对离散特征也启用 `use_not`，离散部分还会追加反值列。

### 2.5 原始交叉验证流程中的数据泄漏

五折交叉验证与 RRL 模型本身没有冲突，问题出在上游 [`get_data_loader`](../models/rrl/experiment.py) 的执行顺序：代码先在全部数据上调用 `db_enc.fit(X_df, y_df)`，再以 `keep_stat=True` 计算全部数据的连续特征均值和标准差，完成整体变换后才调用 `KFold` 划分训练折和测试折。于是某轮测试折的特征分布已经参与训练输入的标准化。泄露的主要是测试折的均值、标准差、缺失值分布和离散类别集合，而不是特征与标签的对应关系；`label_enc` 看到 0、1 两个类别的名称不是主要问题。

严格流程必须先生成 fold，然后在每一轮新建编码器，只用训练折执行 `fit`，再用同一个编码器分别 `transform` 训练折和测试折。测试折出现训练中未见类别时，还应使用 `handle_unknown="ignore"` 或预先固定合法类别。作业文档只要求 K 折交叉验证，并未要求使用分层版本；源码使用普通 `KFold`，我们针对 DIA 的 24.8% 阳性比例提出 `StratifiedKFold`，这是实验设计改进而非作业原文规定。无论使用哪种 fold，预处理参数都必须只从当轮训练折估计。

## 3. 合取层与析取层

设逻辑层输入为 $x=(x_1,\ldots,x_d)$，其中 $x_i\in\{0,1\}$；包含 $n$ 个节点的逻辑子层使用连接矩阵 $W\in\mathbb R^{d\times n}$。训练参数先按 0.5 阈值离散化：

$$\bar W_{ij}=\mathbf 1[W_{ij}>0.5]$$

$\bar W_{ij}=1$ 表示条件 $x_i$ 被节点 $j$ 选中，$\bar W_{ij}=0$ 表示没有连接。逻辑层中的 $W$ 只决定规则结构，和最终线性层中可正可负、用于衡量类别贡献的权重不同；在离散执行时，0.51 和 0.99 都表示同一条有效连接。

合取节点输出所有已连接条件的逻辑与：

$$y_j^{\land}=\bigwedge_{i:\bar W_{ij}=1}x_i$$

源码用 $r_j=\sum_i(1-x_i)\bar W_{ij}$ 统计“被选中但不成立”的条件数，并令 $y_j^{\land}=1$ 当且仅当 $r_j=0$。析取节点输出已连接条件的逻辑或：

$$y_j^{\lor}=\bigvee_{i:\bar W_{ij}=1}x_i$$

源码用 $r_j=\sum_i x_i\bar W_{ij}$ 统计“被选中且成立”的条件数，并令 $y_j^{\lor}=1$ 当且仅当 $r_j>0$。例如输入 $(1,0,1)$ 且三个条件都被选择时，AND 输出 0，OR 输出 1。对应实现分别位于 [`ConjunctionLayer.binarized_forward`](../models/rrl/rrl/components.py) 和 [`DisjunctionLayer.binarized_forward`](../models/rrl/rrl/components.py)。

[`UnionLayer`](../models/rrl/rrl/components.py) 同时维护相互独立的合取、析取连接矩阵，并把两类输出拼接；参数 $n$ 表示每类节点的数量，所以其输出维度为 $2n$。多层结构可以继续组合上一层规则，例如 $R_1=A\land B$、$R_2=C\lor D$ 后，再构造 $R_3=R_1\lor R_2$；更深层还可以使用上一层规则的反值。没有连接任何输入的空合取恒为 1，空析取恒为 0；始终激活或从不激活的 dead nodes 在规则导出时由 [`extract_rules`](../models/rrl/rrl/components.py) 根据激活次数删除。

## 4. Gradient Grafting：让离散规则能够训练

RRL 真正执行规则时把连接参数二值化为 $\bar W_{ij}=\mathbf 1[W_{ij}>0.5]$。该阶跃函数在阈值处不可导、其余位置导数为 0；如果直接反向传播，优化器无法知道应该把某条连接调高还是调低。为此，每个逻辑层同时计算离散输出 $\bar y=F(\bar W,x)$ 和连续代理输出 $\tilde y=\tilde F(W,x)$：前者严格执行 AND/OR，负责产生模型真正使用的前向结果和损失；后者不代替预测，只提供关于连续连接参数的导数。Gradient Grafting 使用近似梯度：

$$\nabla_W L\approx\frac{\partial L(\bar y)}{\partial\bar y}\frac{\partial\tilde y}{\partial W}$$

第一项来自离散规则造成的真实预测误差，第二项来自连续代理函数，说明改变 $W$ 会怎样影响逻辑节点。若 $\partial L/\partial\bar y=-0.8$、$\partial\tilde y/\partial W=0.3$，则近似梯度为 $-0.24$；梯度下降会增大 $W$，它最终可能由小于 0.5 变成大于 0.5，使该连接真正开启。这不是离散模型不存在的“真实导数”，而是一种保持前向规则精确、为反向优化提供方向的代理梯度。

源码中，[`ConjunctionLayer.forward`](../models/rrl/rrl/components.py) 和 `DisjunctionLayer.forward` 先分别得到 `res_tilde` 与 `res_bar`，再调用 `GradGraft.apply(res_bar, res_tilde)`。自定义 [`GradGraft`](../models/rrl/rrl/components.py) 的 `forward` 返回第一个参数，因此后续网络只看到离散输出；其 `backward` 对第一个参数返回 `None`，却把上游梯度原样传给第二个参数，因此 PyTorch 会沿 `continuous_forward` 计算 $W$ 的梯度。训练循环随后执行 `loss.backward()`、`optimizer.step()`，并把逻辑权重截断到 $[0,1]$。嫁接发生在每个合取、析取子层内部，因而深层网络也是逐层保持“离散前向、连续反向”，这对应当前扩展版代码采用的 Hierarchical Gradient Grafting 思想。

## 5. NLAF：高维条件下的连续逻辑代理

AND/OR 的值不难计算，困难在于“哪些条件参与规则”由 $\bar W_i=\mathbf 1[W_i>0.5]$ 决定，这种离散结构几乎处处导数为 0。RRL 因此用严格的离散 AND/OR 负责训练时的前向结果、测试和规则解释，同时用可导的连续逻辑函数为 Gradient Grafting 提供反向路径。NLAF 全称 Novel Logical Activation Functions（新型逻辑激活函数）；它不是最终的模糊规则，而是相对于原始 LAF 更适合高维输入的连续代理。

设 $x_i$ 表示条件是否成立，$W_i$ 表示该条件被规则选择的程度。三条件 AND 中，每一项必须满足：$W_i=0$ 时无论 $x_i$ 为何都被忽略；$W_i=1,x_i=1$ 时不破坏 AND；$W_i=1,x_i=0$ 时应使 AND 为 0。NLAF 定义

$$\mathcal G(t)=1-\frac{1}{1-(\alpha t)^\beta},\qquad \mathcal P(z)=\frac{1}{(1-z)^\gamma}$$

其中 $\mathcal G(0)=0$，$t$ 接近 1 时 $\mathcal G(t)$ 是绝对值较大的负数；$\alpha<1$ 避免 $t=1$ 时除零，$\beta$ 调节响应向 1 附近集中的程度，$\gamma$ 调节最终投影趋向 0 或 1 的速度。连续合取和析取写成：

$$\widetilde{\operatorname{AND}}(X,W)=\mathcal P\!\left(-\mathcal G(1-X)\mathcal G(W)\right),\qquad \widetilde{\operatorname{OR}}(X,W)=1-\mathcal P\!\left(-\mathcal G(X)\mathcal G(W)\right)$$

对 AND，$\mathcal G(1-x_i)\mathcal G(W_i)$ 衡量“条件不成立的程度 × 条件被选择的程度”：$W_i=0$ 或 $x_i=1$ 时至少一个因子为 0，该条件不产生违反；只有 $W_i\approx1,x_i=0$ 时产生很大的正值，把输出压向 0。对 OR，$\mathcal G(x_i)\mathcal G(W_i)$ 衡量“条件成立的程度 × 条件被选择的程度”：没有已选条件成立时输出为 0，只要出现成立且被选择的条件就趋向 1。因此可以把连续 AND 理解为累计违反证据，把连续 OR 理解为累计成立证据；真正的 `binarized_forward` 仍输出精确的 0/1。

对 $X$ 和 $W$ 同时应用 $\mathcal G$，并不意味着原始输入也要训练。第一层主要对 $W$ 求导，但 $X$ 决定每个样本应给 $W$ 什么训练信号，例如 $\partial\tilde y/\partial W_i$ 中含有 $\mathcal G(1-x_i)$ 或 $\mathcal G(x_i)$。在深层 RRL 中，当前层的 $X$ 又是上一层的输出；为了更新上一层参数，链式法则需要 $\partial L/\partial X$，故连续逻辑函数也必须对输入可导。两边采用同一变换还把每对输入、权重的贡献分解为 $\mathcal G(X_{bi})\mathcal G(W_{ij})$，从而既保持“权重为 0 即排除”的逻辑边界，也能用矩阵乘法实现。只变换一边并非数学上绝对不可能，但会形成另一套代理函数，需要重新验证边界行为、梯度和训练效果。

原始连续 AND 为 $y=\prod_i a_i$，其中 $a_i=1-W_i(1-x_i)\in[0,1]$，故

$$\frac{\partial y}{\partial W_j}=-(1-x_j)\prod_{i\ne j}a_i$$

若另外 99 项平均为 0.9，连乘仅约 $0.9^{99}=2.95\times10^{-5}$；999 项时约为 $1.94\times10^{-46}$，高维输入使梯度几乎消失。NLAF 改用变换后的加和 $S=\sum_i\phi(x_i)\phi(W_i)$ 和有理投影 $(1+S)^{-\gamma}$，求某个 $W_j$ 的梯度时不再连乘所有其他输入。其投影也比指数函数衰减慢：$e^{-10}\approx4.5\times10^{-5}$、$e^{-100}\approx3.7\times10^{-44}$，而 $\gamma=1$ 时 $\mathcal P(-10)=1/11\approx0.091$、$\mathcal P(-100)=1/101\approx0.0099$。这只能缓解而不能保证彻底消除梯度消失。

计算方面，设批量大小为 $B$、输入数为 $d$、逻辑节点数为 $n$。原始函数需要为每个样本、输入和节点计算两两组合，广播产生 $B\times d\times n$ 中间张量；NLAF 的可分离形式可直接计算

$$\underbrace{\mathcal G(X)}_{B\times d}\underbrace{\mathcal G(W)}_{d\times n}=\underbrace{S}_{B\times n}$$

只需保存规模约为 $Bd+dn+Bn$ 的二维矩阵。以 DIA 的 $B=64,d=1610,n=64$ 为例，原始三维中间量有 $6{,}594{,}560$ 个元素，单个 `float32` 张量约 25.2 MiB；矩阵乘法涉及的三个主要矩阵合计约 210,176 个元素，且能直接利用 PyTorch/GPU 的成熟矩阵乘法实现。NLAF 因而相对原始 LAF 更能保留高维梯度、占用更少显存且运行更快，但代价是引入 $\alpha,\beta,\gamma$ 三个超参数。普通 sigmoid 也能参与设计其他软 AND/OR，但它本身只负责压缩数值，仍需额外设计条件选择、随规则长度变化的阈值及空规则行为；斜率大时接近硬逻辑却容易饱和，斜率小时梯度较好但更模糊。替换 NLAF 因而属于需要重新验证的方法改造，而不是本项目复现 RRL 的必要步骤。

## 6. 训练目标：温度、交叉熵与复杂度

最后一个逻辑层产生二值规则向量 $R(x)=(R_1,\ldots,R_m)$，线性层为每个类别计算 logit：$z_c=b_c+\sum_jw_{cj}R_j$。训练时先用正温度 $T$ 缩放 logits，再由交叉熵衡量真实类别 $y$ 的预测误差：

$$p_c=\frac{\exp(z_c/T)}{\sum_k\exp(z_k/T)},\qquad L_{\mathrm{CE}}=-\log p_y$$

源码把 $t=\log T$ 定义为可训练参数并使用 $T=\exp(t)$，因此温度始终为正；`--temp` 设置初始值。$T<1$ 放大类别分数差，使概率更尖锐，$T>1$ 使概率更平缓。正温度不改变 logits 的大小顺序及 `argmax` 预测，却会改变交叉熵和梯度。`CrossEntropyLoss` 已在内部完成 Softmax，故训练代码直接传入 `self.net.forward(X) / exp(t)`。损失梯度依次通过线性层、离散规则输出和 Gradient Grafting 进入连续逻辑代理，最终更新规则连接。

当前源码真正优化的总目标是：

$$L=L_{\mathrm{CE}}+\lambda\left(\sum_{\text{逻辑层}}\lVert W_{\mathrm{logic}}\rVert_2^2+\lVert W_{\mathrm{linear}}\rVert_2^2\right)$$

其中 `-wd/--weight_decay` 是 $\lambda$，Adam 自身的 `weight_decay` 被设为 0，正则项由代码手动加入。逻辑权重被限制在 $[0,1]$，L2 将不重要的连接推向 0，使它们可能跌到离散连接阈值 0.5 以下，因而能间接简化规则；但它惩罚的是连续权重而非硬边数，也同时收缩线性分类权重。默认 `weight_decay=0`，所以默认运行没有这项复杂度约束。

[`edge_penalty`](../models/rrl/rrl/models.py) 直接统计 $\sum_{ij}\mathbf 1[W_{ij}>0.5]$，但训练循环只把它写入 TensorBoard，并未加入 loss；已定义的 `l1_penalty()` 和 `mixed_penalty()` 同样未被使用。当前实际控制模型复杂度的是层数、每层节点数、非零 L2 系数，以及导出时剔除无效节点。DIA 阳性约占 24.8%，但当前 `CrossEntropyLoss` 没有类别权重；命令行虽然解析 `--weighted`，该参数没有被传入或使用，实际上不生效。是否引入加权损失应在训练折内作为实验选择，并结合 macro-F1、阳性 precision/recall/F1、PR-AUC 和混淆矩阵判断，而不能误以为原参数已经处理类别不平衡。

## 7. 从连接矩阵还原可读规则

规则提取不参与训练，而是在模型训练结束后把连接矩阵和线性权重翻译成人类可读的逻辑表达式。源码先在训练数据上重新执行严格的离散前向传播，记录规则 $R_j$ 在 $N$ 个样本中输出 1 的次数 $n_j$。支持度（Support）定义为：

$$\operatorname{Support}(R_j)=\frac{n_j}{N}$$

Support 就是规则的覆盖率：0 表示没有训练样本满足，0.18 表示覆盖 18%，1 表示全部满足；它只衡量出现频率，不是分类置信度，也不表示规则成立时阳性所占比例。后者应另算 $P(Y=1\mid R=1)=\#(R=1,Y=1)/\#(R=1)$，当前规则表没有提供。Support 与线性权重应结合阅读：高权重、极低 Support 的规则可能只影响极少样本，而覆盖较广的规则具有不同的实际意义。

激活次数还用于识别 dead nodes。$n_j=0$ 的规则在训练集上贡献恒为 0，可以从报告剔除；$n_j=N$ 的规则贡献恒为 $w_{cj}$，可折叠进类别偏置 $b'_c=b_c+\sum_{j:R_j\equiv1}w_{cj}$。这里的恒真、恒假是相对于用于统计的训练数据而言：某规则可能只因训练数据范围有限而从未激活，规则打印仅简化报告，并未从模型中物理删除节点。

[`rule_print`](../models/rrl/rrl/models.py) 随后按 $W_{ij}>0.5$ 恢复每个节点选中的输入；合取节点用 AND 连接，析取节点用 OR 连接，NOT 和 skip connection 也按来源保留。二值化层的标准化阈值通过 $c_{\mathrm{raw}}=cs_{\mathrm{train}}+\mu_{\mathrm{train}}$ 还原到原始单位。对同一特征的冗余阈值会按逻辑等价关系合并，例如 $(x>200)\land(x>300)\equiv x>300$、$(x>200)\lor(x>300)\equiv x>200$；方向相反的上下界不能随意删除，因为它们可能表达区间。条件组合完全相同的节点映射为同一规则 ID，其线性权重相加。多层结构则递归展开并用括号保留运算顺序，例如 $R_1=A\land B$、$R_2=C\lor D$、$R_3=R_1\lor\neg R_2$ 最终显示为 $(A\land B)\lor\neg(C\lor D)$。

最终表格包含规则 ID、各类别线性权重、Support 和规则文本。二分类时仍应使用 $\Delta w_j=w_{\mathrm{positive},j}-w_{\mathrm{negative},j}$ 判断相对支持方向，而不能只看阳性权重是否为正。规则结构直接读取模型实际使用的离散连接，因而核心解释与模型同源；但 Support 和 dead-node 简化依赖所提供的数据范围，应把输出理解为“训练数据范围内离散模型的紧凑说明”。

## 8. DIA 的分类评估指标

DIA 合并后有 597 个样本，其中阴性 449 个（75.2%）、阳性 148 个（24.8%）。若模型把所有样本都预测为阴性，Accuracy 仍有 $449/597\approx75.2\%$，却完全无法识别阳性，因此 Accuracy 只能作为辅助指标。设 TP、FP、FN、TN 分别表示阳性判对、阴性误报为阳性、阳性漏报为阴性、阴性判对，则 $\operatorname{Accuracy}=(TP+TN)/N$；混淆矩阵应与汇总分数同时报告，以显示错误究竟来自误报还是漏报。

把阳性视为目标类时，Precision、Recall 和 F1 为：

$$P_+=\frac{TP}{TP+FP},\qquad R_+=\frac{TP}{TP+FN},\qquad F1_+=\frac{2P_+R_+}{P_++R_+}$$

$P_+$ 回答“预测为阳性的样本中多少是真的”，$R_+$（Sensitivity）回答“实际阳性中找回了多少”，$F1_+$ 是二者的调和平均；二分类报告中未加说明的 `F1` 通常就是阳性 F1，但名称具有歧义，本文统一显式写 `Positive F1`。把阴性视为目标类时，$P_-=TN/(TN+FN)$、$R_-=TN/(TN+FP)$，后者也称 Specificity，$F1_-=2P_-\cdot R_-/(P_-+R_-)$。Macro-F1 对两个类别等权平均：

$$F1_{\mathrm{macro}}=\frac{F1_++F1_-}{2}$$

始终预测阴性的模型有 $F1_+=0$、$F1_-\approx0.858$，故 Macro-F1 约为 0.429，能够揭示 75.2% Accuracy 掩盖的少数类失败。Positive Recall 对 DIA 风险漏检尤其直观，Positive F1 衡量固定阈值下阳性查准与查全的平衡，Macro-F1 则防止多数类主导总体评价。

模型先为每个样本产生连续阳性分数，再以阈值得到类别。ROC 曲线遍历所有阈值，以 $\mathrm{FPR}=FP/(FP+TN)$ 为横轴、$\mathrm{TPR}=TP/(TP+FN)$ 为纵轴；ROC-AUC 是曲线面积，并等价于随机抽取一名阳性和一名阴性时，阳性分数高于阴性分数的概率（同分通常计半次）。例如阳性分数为 0.90、0.60，阴性分数为 0.70、0.20，四个正负配对中三次把阳性排在前面，故 ROC-AUC 为 $3/4=0.75$。0.5 表示接近随机排序，1 表示所有阳性均排在阴性之前。ROC-AUC 衡量跨阈值排序而非概率校准或某个工作阈值的分类效果；AUC 高的模型仍可能因阈值选择不当而得到很差的 F1。

PR 曲线同样遍历阈值，但以 Recall 为横轴、Precision 为纵轴；PR-AUC 概括“逐步找回更多阳性时能否保持较高的阳性预测纯度”。它与 Positive F1 的区别是：F1 只描述一个阈值对应的 PR 曲线工作点，PR-AUC/AP 则考察全部阈值下的阳性排序质量。类别不平衡时，ROC 的 FPR 分母含大量阴性，可能弱化误报的实际负担：若有 100 个阳性、10,000 个阴性，模型得到 $TP=80,FP=200$，则 Recall 为 0.80、FPR 仅 0.02，看似很好，但 Precision 只有 $80/(80+200)\approx0.286$；PR 指标会直接暴露“多数阳性警报其实是误报”。DIA 的不平衡较温和但仍存在同类问题，因此 PR-AUC 是 ROC-AUC 的有价值补充，而不是替代品。

随机排序的预期 Precision 约等于阳性比例，故 DIA 的 PR 参考基线约为 $148/597\approx0.248$；这一基线随类别比例变化，不能直接比较阳性比例不同的数据集。实践中常报告 Average Precision（AP），它与采用梯形插值计算的几何 PR-AUC 略有差异，报告时必须明确名称和实现。总体而言，Accuracy 回答固定阈值下的总体正确率，Positive F1 回答该阈值下阳性 Precision/Recall 的平衡，Macro-F1 检查两类是否均衡，ROC-AUC 衡量正负样本的整体排序，PR-AUC/AP 衡量找回阳性时的误报代价。DIA 建议以 Macro-F1 为主要模型选择指标，同时报告 Positive F1、Positive Recall、ROC-AUC、AP 和混淆矩阵；Accuracy 仅作常规参考。

五折分层交叉验证中，每折必须只用训练折拟合全部预处理和模型，在留出折计算指标，最后报告五折均值与标准差；拼接每个样本恰好一次的 out-of-fold 连续分数，可以绘制总体 ROC/PR 曲线并汇总混淆矩阵。留出折不能用于选择预处理阈值、模型结构、超参数或最佳 epoch，这些选择只能在当前训练折内部完成。现有 [`test`](../models/rrl/rrl/models.py) 只计算 Accuracy、Macro-F1、混淆矩阵和分类报告；后续实验管线还需保留连续阳性分数并补充 Positive F1、ROC-AUC、AP 以及跨折汇总。

## 9. 当前理解与后续内容

目前已经明确从输入编码、规则训练、训练目标、规则提取到分类评估的完整链路，也识别了交叉验证预处理泄漏、默认复杂度约束为空、`--weighted` 无效和评估指标不足等源码问题。下一阶段可以从讲解转入 DIA 的可复现实验设计与最小实现；只有在完成讲解并得到确认后才会继续写入本文。

## 参考资料

[NeurIPS 2021 论文](https://proceedings.neurips.cc/paper/2021/file/ffbd6cbb019a1413183c8d08f2929307-Paper.pdf)；[TPAMI 扩展版本](https://arxiv.org/abs/2310.14336)；[本地 RRL README](../models/rrl/README.md)；[scikit-learn 数据泄漏说明](https://scikit-learn.org/1.8/common_pitfalls.html#data-leakage)。
