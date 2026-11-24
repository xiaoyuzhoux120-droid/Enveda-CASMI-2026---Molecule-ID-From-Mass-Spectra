# 方向 4：候选检索、指纹预测、谱图嵌入、重排序与集成

日期：2026-11-24 · 范围：研究和 CPU 小实验，没有向 Kaggle 提交，也没有改动 main。

## 0. 结论摘要

1. **比较基准**：当前真实最佳是 Kaggle notebook `xiaoyuzhoux120/casmi26-nightly-frp5-20261119`（公开榜 0.495，提交号 58552147）。GitHub 仓库 main 只是 2026-09 的旧快照（0.176），不能代表现状。本报告的建议以 FRP5 为基准。
2. **FRP5 本质上是一个多层排名融合系统**，由十余个公开 notebook 的组件叠成（见 §1）。从 0.474 到 0.495 的爬升，每一步大多只差 +0.001 到 +0.008。
3. **公开榜噪声**：公开榜约 120 个分子（测试集的 30%）。按模拟，同一方案公开榜 MRR 的标准误约 0.036；两个只改动 10 到 40 个列表的方案，其公开分差的标准差为 0.006 到 0.012（§3）。也就是说，最近多数夜间提交之间的差异处在噪声范围内。最终两次提交的选择不应只依据公开榜。
4. **本线程实验（类别 2 代理，独立开发集加冻结后的留出集）**：
   - 对于结构在 COCONUT 中、但谱库里没有它自身谱图的分子，用**全局类似物检索**（不限质量；碎片 cosine 加中性丢失 cosine）选出邻居，再按邻居结构的 Tanimoto 给候选排序。候选池内 MRR 从 0.123 提升到 **0.396**（留出集，+0.273，配对 bootstrap CI95 [0.242, 0.300]）。基线是只用同分异构体邻居（35 ppm 窗口内）的历史 hybrid 写法。开发集结果几乎相同（0.117 → 0.392）。
   - 中性丢失通道相对只用碎片 cosine：开发集 +0.012，留出集 +0.041。
   - 把同分异构体邻居通道和类似物通道等权融合（z-score 或 RRF），结果**反而更差**（0.31 / 0.29 对 0.40）。弱通道等权加入会稀释强通道。
   - COCONUT `annotation_level` 先验在开发集上 +0.024（CI 下限 0.0002），在留出集上只有 +0.006（CI [-0.019, 0.028]），**未复现**，不建议采用。
   - **与 FRP5 的关系**：FRP5 的 Second engine 已经实现了全局类似物检索（±200 Da、modified/shifted cosine、entropy 加权、top-80 类似物），并把 `max(Tanimoto·w³)` 作为 GBM 特征。所以这个实验是对现有设计的**独立验证**，不是新发现。它可以直接落地的部分是：清点所有仍使用"同分异构体邻居"或等权融合弱通道的层（§4）。
5. **文献结论**（§5）：MassSpecGym 上的对比学习/回归对齐方法（JESTR、SpecBridge）和生成式重排（GLMR）报告的提升很大，但都依赖 A100 级 GPU 和 Transformer 预训练（DreaMS、ChemBERTa）。在剩余 20 天、T4 推理 9 小时限制下，性价比最高的仍是**改进融合与验证**，而不是训练新的大型编码器。

## 1. FRP5 0.495 的检索、重排、集成结构（据 notebook 源码）

| 层 | 作用 | 来源 |
|---|---|---|
| Main engine | 10 ppm（空时 30 ppm）质量窗口候选，加上"修饰谱图类似物"生成的结构；FPNet 指纹模型和 4 个 LightGBM 排序；`z(ranker)+0.15·pop` | 0.465 基础 pipeline（base64 嵌入） |
| PubChem channel | FPNet A+B 和 fold-safe checkpoint 在 PubChem 层筛选，`S=z(fp)+0.15·pop`，满足条件时提升到第 1 位 | 同上 |
| Second engine | 训练结构、COCONUT、ChEBI、LIPID MAPS、PubChemLite；谱库匹配、**全局类似物传播**、指纹、碎片匹配；两组深度为 6 的 GBM 平均 | 同上 |
| GLACIER/ICEBERG | 正向谱图模拟重排（同分子式），NH4/失水加合物代理 | 同上 |
| 神经打分器融合 | public 和 ours 两个模型 0.5·z 平均；CUSTOM：rank1 用 w_top=0.25，rank2–25 用 w_mid=0.75，RRF k=3 | Spectral Parallax |
| Variant M | 碎片检查改 rank1；`1/(3+r_base)+0.75/(3+r_nn)+0.12·pop/20+0.12·lib·s+0.12·np` | busyaprime 等 |
| v8 MASS | 首位质量误差 >5 ppm 时与 rank2/3 交换 | romanrozen |
| 邻居网络 D-L1 | 测试分子之间 modified cosine ≥0.6 互相传播证据，λ=0.35，加训练集已知节点 | 本账号 |
| 外部谱库 overlay | cosine ≥0.85 且 ≥4 个匹配峰的外库命中直接放到第 1 位 | 本账号重建的 extlib |
| FR 3-25 / P5 tail | 学习型碎片重排第 3–25 位；Engine 1 新候选插入尾部 | 本账号 |

观察：

- 系统一共约 8 层融合，每层都是手工设定的系数（0.25/0.75/0.12/0.35/k=3…），几乎全部是**按公开榜挑选**的。
- notebook 自己的记录也显示，离线指标多次误判榜单（自训 FPNet 的 Tanimoto 提升 26.5%，7 次替换在榜上全部下降），而且 rank1 是得分和失分的主要位置。

## 2. 本线程实验：类别 2 候选重排

代码：`research/retrieval_rerank/analog_rerank_experiment.py`；结果：`research/retrieval_rerank/results/`。

**协议**

- 参考库：train 分区（`casmi_ml.data.split_key` 的 SHA256 分区）分子，每个分子最多 3 条谱，共 603,591 条。
- 查询：dev 分区（用于选择）或 holdout 分区（用于确认）中 InChIKey14 出现在 COCONUT 的分子，各取按哈希顺序的前 600 个；每分子最多 4 条 [M+H]+/[M−H]− 谱。查询结构本身没有参考谱，模拟类别 2。
- 候选池：COCONUT（738,827）加训练结构，取中性质量 ±max(15 ppm, 0.004 Da)；中位数 57–64 个候选，真结构入池率 93–94%。
- 指标：池内 MRR@25，按数据集的 InChIKey14 计算（未复现官方的 tautomer 规范化）；同分时按悲观方式计名次；配对 bootstrap 1000 次。
- 留出集运行前已把方案冻结到 `results/frozen_selection.json`。

**结果（池内 MRR@25）**

| 方法 | dev (n=565) | holdout (n=558) | holdout 相对历史写法的差值 CI95 |
|---|---:|---:|---|
| iso_hist：同分异构体邻居，max(T·(0.5+0.5s))（历史 hybrid） | 0.117 | 0.123 | — |
| frag_hist：全局类似物，只用碎片 cosine | 0.379 | 0.355 | [0.203, 0.263] |
| **ana_hist：全局类似物，碎片加中性丢失** | **0.392** | **0.396** | **[0.242, 0.300]** |
| ana_knn：Σ s⁴·T 加权投票 | 0.372 | 0.374 | [0.221, 0.280] |
| z(ana_knn)+0.25·z(annotation 先验)（开发集首选） | 0.415 | 0.402 | [0.249, 0.307] |
| z(iso)+z(ana) 等权融合 | 0.310 | 0.312 | — |
| RRF(k=3) iso+ana | 0.273 | 0.287 | — |
| 只用 annotation 先验 | 0.195 | 0.198 | — |

- 首选方案对 ana_hist 的差值：dev +0.024 [0.000, 0.044]，holdout +0.006 [−0.019, 0.028]。冻结标准（CI 下限 >0）对"先验带来增益"这一项**未通过**。结论应写成"全局类似物邻居有效；先验没有可复现的增益"。
- 敏感性分析：剔除最近邻与真结构 Tanimoto=1 的情形（约 4%，可能是盐型、立体或互变异构体重复）后，结论不变（holdout ana_hist 0.394，iso_hist 0.117）。
- 计算量：CPU 4 核，构建参考库约 2 分钟，600 个查询约 2.5 分钟；峰值内存约 5 GB。

**局限**

- 查询谱多为 Orbitrap/QTOF（COCONUT 中的非训练分子只有 57 个有 timsTOF 谱），隐藏测试集则全部是 timsTOF。
- 代理只覆盖类别 2，不能换算成榜单分数。
- 历史 hybrid 写法是一个弱对照，FRP5 已经不再使用它。

## 3. 公开榜噪声与最终提交选择

模拟设定：400 个分子，公开 120 个，系统 MRR 约 0.49（逐分子倒数名次分布按 rank1 45%、0 分 40% 构造）。

- 同一方案公开榜 MRR 的标准误约 0.036；公开与私有分数之差的标准差约 0.051。
- 两个方案改动 m 个列表、每个改动为 rank1↔rank2（±0.5）、净改善 n 个时：

| m | 净改善 | 全测试集真实差 | 公开榜差的 SD | 公开榜显示 ≤0 的概率 |
|---:|---:|---:|---:|---:|
| 10 | 2 | +0.0025 | 0.0060 | 0.47 |
| 20 | 4 | +0.0050 | 0.0085 | 0.37 |
| 40 | 8 | +0.0100 | 0.0121 | 0.25 |

含义：

1. 0.487→0.495 这样的爬升大约相当于 2 个分子；公开榜很难区分。
2. 选最终提交时，建议一份选"公开榜最高且改动最少"的版本，另一份选"在离线面板（例如已附加的 natural-product-panel-benchmark，或类别分层的 train 代理）上配对检验最优"的版本，而不是两份都取公开榜前二。

## 4. 建议（按优先级）

| 优先级 | 动作 | 预期 | 成本 |
|---|---|---|---|
| P0 | 建立固定的**类别分层离线面板**：类别 1（库内）、类别 2（COCONUT/PubChem 在池但无谱，用本实验协议）、类别 3（不在池）。每个融合系数的改动都同时在面板上做配对 bootstrap，报告 rank1 变化数 | 降低按公开榜过拟合的风险 | CPU 数小时；可复用本线程脚本 |
| P1 | 清点 FRP5 中是否仍有通道只用同分异构体邻居，或以较大权重融合弱通道（待核实，例如 Main engine 的库匹配特征、邻居网络已知节点的选取方式）。本实验中，弱通道等权融合使 MRR 从约 0.40 降到 0.31 | 中 | 读代码加面板验证 |
| P1 | 融合方式：用面板上的**学习式融合**（例如对各层得分做 LambdaMART 或逻辑回归，按分子分组交叉验证）代替手工 RRF 系数，并且只对第 2–25 位生效，第 1 位保持保护 | 小到中；notebook 记录显示动 rank1 风险高 | CPU |
| P2 | 全局类似物检索加入 **MS2DeepScore/Spec2Vec 嵌入相似度**作为第二个邻居通道（MS2Query 的做法） | 不确定；MS2Query 报告类似物检索提升明显，但需要独立环境 | CPU 推理可行，需要单独的 venv |
| P3 | 对比/回归对齐编码器（JESTR、SpecBridge 路线） | 文献增益大，但需要 GPU 训练和预训练权重，剩余时间不足 | 高 |

不建议：COCONUT annotation_level 先验（留出集未复现）；在公开榜上继续扫系数。

## 5. 文献与公开实现（已核对来源）

- MassSpecGym（Bushuiev 等，NeurIPS 2024 D&B）：https://arxiv.org/abs/2410.23326 。检索 mass 挑战 hit@1：Random 0.37、Fingerprint FFN 2.54、DeepSets+Fourier 5.24、MIST 14.64；代码 https://github.com/pluskal-lab/MassSpecGym 。
- JESTR（Kalia 等）：分子 GNN 与谱图 MLP 的对比学习联合嵌入；MassSpecGym mass hit@1 15.13（no-reg 15.62）；在 A100 上训练。https://arxiv.org/abs/2411.14464 ；https://github.com/HassounLab/JESTR
- SpecBridge（Wang 等，2026）：DreaMS 编码器回归到冻结的 ChemBERTa-2 嵌入；论文报告 MassSpecGym R@1 84.7、MRR 88.1（其 Random 行与官方榜不一致，协议需要核实）。https://arxiv.org/abs/2601.17204 ；https://github.com/HassounLab/SpecBridge
- GLMR（Zhang 等，2025）：先用对比检索取 top-40，再条件生成 SMILES 进行重排；报告 R@1 64–68；没有公开训练代码。https://arxiv.org/abs/2511.06259
- MS2Query（de Jonge 等，Nat Commun 2023）：用 MS2DeepScore、Spec2Vec 和质量差做随机森林的类似物检索，与本实验的"全局类似物邻居"思路一致。https://www.nature.com/articles/s41467-023-37446-4
- MS2DeepScore：https://pmc.ncbi.nlm.nih.gov/articles/PMC8556919/ ；DreaMS（Nat Biotechnol 2025）：https://www.nature.com/articles/s41587-025-02663-3
- PubChemLite 与 MetFrag（Schymanski 等，J Cheminform 2021），说明参考文献/专利计数先验的作用：https://doi.org/10.1186/s13321-021-00489-0
- ICEBERG（Goldman 等）正向模拟：https://arxiv.org/abs/2304.13136
- 比赛信息（二手来源，规则以 Kaggle 页面为准）：约 400 个测试分子、全部为 timsTOF；MRR@25 按 RDKit tautomer 规范化后的 InChIKey14 计算，RDKit 固定为 2026.03.3；代码赛，离线运行，9 小时；队伍合并截止 2026-12-07，比赛结束 2026-12-14。来源：Business Wire 新闻稿（2026-09-15），以及公开参赛者 README，例如 https://raw.githubusercontent.com/mxm-1x/Enveda-CASMI-2026/main/README.md 。

## 6. 复现

```bash
.venv/bin/kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data && (cd data && unzip -o *.zip)
.venv/bin/kaggle datasets download -d aidensong123/casmi26-coconut-202609 -p external/coconut_dl --unzip
.venv/bin/python research/retrieval_rerank/analog_rerank_experiment.py --split dev --n 600
.venv/bin/python research/retrieval_rerank/analog_rerank_experiment.py --split holdout --n 600
```

dev 运行时 `iso_knn` 存在除零 bug（全部邻居得分为 0 时），留出集运行前已修复。dev 表中 `iso_knn` 和 `z_isoknn+…` 两项无效，不参与选择。
