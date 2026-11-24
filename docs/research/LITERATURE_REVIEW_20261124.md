# 文献与历届 CASMI 方法调研（方向 5）

日期：2026-11-24。范围：近年 MS/MS 分子识别论文、公开实现、历届 CASMI 优胜方法、本次 Kaggle 比赛公开方案，以及它们与比赛外部数据规则的合规性。

**阅读说明**：本文所有数字均来自原文或公开仓库，没有本地复现。多数论文网页在抓取时被截断，后半部分（通常是结果表）来自抓取工具对未读部分的摘要，标为“（摘要，待核）”。标注“已读原文”的数字来自逐字读到的文本。没有一个数字是我们自己跑出来的。

**背景校正**：仓库 `main` 只到 0.176 的历史快照。按协调线程转达，你目前最好的公开分是 **0.495**（Notebook `xiaoyuzhoux120/casmi26-nightly-frp5-20261119`：entropy 谱库检索 + COCONUT/PubChem/LOTUS 候选池 + GLACIER/ICEBERG 模拟谱 + 双神经打分器 RRF + MetFrag-lite + reranker + 邻居网络），榜首约 **0.554**。我没有读到这个 Notebook 的代码，因此下面的建议是“对照文献，哪些成分你可能还没有”，需要你对照 Notebook 确认。

---

## 1. 结论先行：最可能缩小 0.495 → 0.554 差距的方向

按“预期收益 / 成本 / 合规风险”排序：

| 优先级 | 方向 | 文献证据 | 你的管线可能缺的部分 | 成本 | 合规 |
|---|---|---|---|---|---|
| P0 | **合规审计 GLACIER/ICEBERG 权重** | ICEBERG 的 NIST 训练权重是单独门控发放的（FRIGID issue #10 原话转述）；NIST 及其衍生权重被主办方明确禁止 | 确认你用的是 NPLIB1/MassSpecGym 公开数据训练的 checkpoint，而不是 NIST 版本 | 低 | **高风险，若用错会被取消资格** |
| P1 | **候选内 listwise / 同质量硬负样本训练的打分器** | 公开仓库 christianmcb：FPNet（peak transformer，10,226 位多族指纹 logits，用同质量硬负样本训练排序）把公开分从 0.146 提到 0.229；MassSpecGym 上 MIST 在同条件下 hit@1 14.64% 远超 fingerprint FFN 2.54% | 若双神经打分器是按位 BCE 训练的指纹预测，改为“候选内 softmax/InfoNCE + 同质量负样本”是最直接的升级 | 中（GPU 数小时） | 低 |
| P1 | **分子式推断作为候选过滤/特征** | "MassSpecGym in the Wild" 表 2（已读原文）：mass 赛道随机基线 hit@1 0.43%，给定真实分子式 12.98%；MIST + 数据安全的 MIST-CF 分子式 32.14%，给真实分子式 52.84% | 用碎片子式一致性（subformula tree）给候选打分，比仅靠 35 ppm 精确质量窗口强得多 | 中（可自写分解，免 SIRIUS） | 低（MIST-CF 公开权重只用 NPLIB1） |
| P1 | **CASMI 式多证据等权集成 + 文献热度先验** | CASMI 2016（摘要，待核）：6 个碎片打分器等权组合 Top1 82 vs 最佳单模型 70；Category 3 只用“参考文献计数”就 Top1 143/208；christianmcb：PubChem 热度先验 w=0.15 带来 +0.018 公开分；"in the Wild" 表 3：PubChem 默认排序（CID 顺序）单独就能 hit@1 49.98% | 若没有热度先验（PubChem 记录数/PubMed 数/CID），这是低成本高收益项 | 低 | 低（PubChem 公共领域；LOTUS CC BY；NPAtlas CC BY-NC 允许） |
| P2 | **DreaMS 预训练谱编码器 + 冻结分子编码器对齐** | SpecBridge（摘要，待核）：DreaMS + 冻结 ChemBERTa-2 + 残差映射 MLP，MassSpecGym R@1 84.7（远高于 GLMR 68.5）；但 "in the Wild" 指出候选规范化伪影可把 DreaMS+ChemBERTa 抬到 82.41%，修正后只有 6.70% | DreaMS 嵌入（1024 维）作为第三路打分器或 reranker 特征 | 中（推理需 GPU 或较慢 CPU） | 中：GeMS 来自 GNPS/MassIVE 公开数据；代码 MIT（README 徽章）；权重许可未核实 |
| P2 | **模拟谱预训练 + candidate-aware MaxSim** | "Separability Is Not Alignability"（NeurIPS 2026 workshop）：用 GLACIER/FraGNNet 对 5.9M 分子模拟谱预训练全新编码器 + 候选感知 MaxSim，MassSpecGym MRR 0.566、R@1 0.425；冻结 DreaMS 对比学习只有 MRR 0.227 | 你已有 GLACIER/ICEBERG 模拟谱，可把它们用于**编码器预训练**，而不仅是模拟谱库匹配 | 高（他们用 4×A100） | 取决于模拟器权重来源（见 P0） |
| P2 | **谱图类似物传播（analog propagation）特征进 LambdaRank** | eganeganegan：±200 Da 窗口找谱图近邻，Tanimoto × 谱相似度⁴ 传播到精确质量候选；加 12 个类似物特征后公开分 0.128 → 0.185，再加 12 个 entropy 通道特征到 0.226 | 你的“邻居网络”可能已覆盖；可检查是否把类似物证据作为 reranker 特征（而非仅 RRF） | 低 | 低 |
| P3 | **de novo 生成（class 3 新分子）** | 修正后 MassSpecGym top-1：FRIGID 9.05%（全 InChIKey 匹配 4.97%），DualLGD 23.89%（作者自报，未经第三方核验），DiffMS 2.30%，MBGen 7.58%；GEMS 在已知分子式下 hit@1 30.22% 但代码未发布 | 只对“库里没有正确结构”的查询追加生成候选；收益上限受 class 3 比例限制 | 高 | FRIGID checkpoint 许可不清（Zenodo 写 CC BY-NC，代码 Apache-2.0，issue 未回复）；其 MIST 编码器用 GLACIER 模拟谱增强，需同 P0 审计 |

我的判断：在剩下的约 20 天里（最终提交截止 2026-12-14 23:59 UTC），P0 必须先做；P1 三项都便宜且证据最直接；P2 选一项；P3 不建议作为主线。

---

## 2. 本次 Kaggle 比赛的公开方案（2026-11-24 读取）

比赛未结束，没有获奖方案公开。以下是 GitHub 上公开的参赛仓库，数字由作者自报，我们未复现。

| 仓库 | 方法 | 自报公开分 | 外部数据 / 模型 |
|---|---|---|---|
| [christianmcb/enveda-casmi26](https://github.com/christianmcb/enveda-casmi26) | 谱库检索 + 学习型指纹排序；换用公开 FPNet 后提升最大；PubChem 热度先验 | 0.076 → 0.128（+COCONUT）→ 0.146（+热度先验）→ 0.229（FPNet） | COCONUT、PubChem 热度先验（数据集 `dmitriigluzdov/casmi26-pubchem-popularity-prior`，含 LOTUS CC BY 4.0 与 NPAtlas CC BY-NC 4.0）、FPNet（Kaggle 用户 ahmedberatozer 发布，作者称其 checkpoint 训练用了所有可验证分子，所以本地无法验证） |
| [eganeganegan/casmi26](https://github.com/eganeganegan/casmi26) | 谱库检索 + 类似物传播 + MetFrag-lite 碎片特征 + LightGBM LambdaRank（最多 73 个特征） | 0.116 → 0.185（类似物特征）→ 0.226（entropy 通道特征） | COCONUT、PubChemLite、ChEMBL 37（CC BY-SA 3.0）；参考了公开 Apache-2.0 引擎“haideptry”（公开分 0.339） |
| [frapercan/casmi26](https://github.com/frapercan/casmi26) | JEPA 预训练的 peak Transformer，InfoNCE/指纹多头，候选用冻结 ChemBERTa-77M-MTR 嵌入 | 未报 | COCONUT、PubChem 子集、PubChemLite、ChemBERTa |
| [sinhaaditya5/kaggle-enveda-casmi26](https://github.com/sinhaaditya5/kaggle-enveda-casmi26) | 质量窗口 cosine 谱库匹配 | 未报 | 无 |
| [corentinlaudicina/enveda-casmi-2026](https://github.com/corentinlaudicina/enveda-casmi-2026) | MLP 预测 2048 位 Morgan，soft Tanimoto 排序，32 个同质量负样本 | 0.098 → 0.112 | COCONUT、PubChem、DreaMS |

**从这些仓库能得到的信息**

- 公开仓库最高 0.229，而你是 0.495，所以这些仓库更多是“组件参考”，不是可以直接照搬的方案。
- 多个仓库把测试集描述为三类：class 1 = 结构在公开谱库中有谱；class 2 = 结构在 PubChem/COCONUT 中但没有谱；class 3 = 新分子。christianmcb 估计“任何候选池只能覆盖约 25–33% 的隐藏分子”，corentinlaudicina 写“约 9/10 的隐藏分子在训练集中没有谱”。这些是参赛者推断，不是主办方数据，但与你此前“unknown 候选召回约 9–12%”的本地诊断方向一致：**候选覆盖率是上限，排序器再好也救不回池子里没有的结构**。
- 公开引擎“haideptry 0.339”和 FPNet 都以 Kaggle 数据集形式流传。若你的 Notebook 引用了任何他人 Kaggle 数据集或权重，需要确认其训练数据来源（见第 5 节）。

---

## 3. 近年论文（按对本赛的用途分组）

### 3.1 基准与评估陷阱（最该先读）

**MassSpecGym**（Bushuiev et al., NeurIPS 2024 Datasets & Benchmarks；[arXiv 2410.23326](https://arxiv.org/abs/2410.23326)；代码 [pluskal-lab/MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)）
- 已读原文：231K 谱、29K 结构；约 10K 分子来自新的 MSnLib；用 MCES < 10 的聚类划分，指出按 2D InChIKey 划分会把 Tanimoto > 0.85 的近似物泄漏进测试集。
- 检索结果（摘要，待核；hit@1/5/20 %）：mass 候选下 Random 0.37/2.01/8.22，DeepSets 1.47/6.21/19.23，Fingerprint FFN 2.54/7.59/20.00，DeepSets+Fourier 5.24/12.58/28.21，**MIST 14.64/34.87/59.15**。分子式候选下 MIST 9.57/22.11/41.12。
- 数据来源 MoNA、MassBank、GNPS 与自测数据（摘要，待核），**不含 NIST**。
- 对本赛的意义：MIST 这类“峰的分子式子树 + Transformer”的编码器比直接对分箱谱做 MLP 强得多；本地验证应采用结构（MCES/Tanimoto）不相交的划分。

**MassSpecGym in the Wild: Uncovering and Correcting Evaluation Pitfalls**（[arXiv 2606.19624](https://arxiv.org/abs/2606.19624)，作者含 Enveda Biosciences）
- 已读原文：首年 26 篇报告 MassSpecGym 结果的论文中至少 18 篇有评估问题，分三类：数据泄漏、捷径学习、实现 bug 与指标不一致。
- 表 1（已读原文）：MIST 不做增强 hit@1 9.57%；无泄漏的 GLACIER 增强 40.31%；有泄漏的设置 47.01%。**模拟谱增强本身是大幅有效的**，前提是增强分子不含测试分子。
- 表 2（已读原文）：分子式信息的价值见第 1 节。
- 表 3（已读原文）：候选 SMILES 规范化不一致会产生捷径，DreaMS+ChemBERTa 被抬到 82.41%，修正后 6.70%；PubChem 默认排序单独 hit@1 49.98%。
- MIST 实现 bug（摘要，待核）：注意力把布尔 mask 直接相加，padding 没被屏蔽。若你用 MIST 代码训练，需检查。
- 建议（摘要部分待核）：预训练排除测试 InChIKey，泛化结论再排除 Tanimoto ≥ 0.7 的类似物；所有 SMILES 统一用 RDKit 规范化；用常数输入谱做“盲谱”测试检查捷径；报告 bootstrap 置信区间。
- 对本赛的直接启示：(1) 检查你的候选池里不同来源的 SMILES 是否统一规范化，否则 reranker 可能学到“来源/写法”捷径，在隐藏集上失效；(2) 把“盲谱”基线加入验证，量化热度先验之类非谱证据的贡献；(3) 热度/CID 先验在本赛是合法的公开信息，可以用，但要清楚它对 class 3 新分子无帮助。

### 3.2 谱 → 指纹 / 嵌入检索

**MIST**（Goldman et al., Nature Machine Intelligence 2023；[论文](https://www.nature.com/articles/s42256-023-00708-3)；[代码](https://github.com/samgoldman97/mist)）
- 为每个峰分配分子式（subformula），在峰分子式集合上用 Transformer 预测指纹。
- README（已读）：主要基准数据 “csi2022” 含 GNPS、**NIST**、MoNA，不能再分发；另有公开的 canopus_train（= NPLIB1）。README 没有说明快速入门脚本下载的权重用哪套数据训练。**结论：不要直接使用 MIST 官方下载的权重，除非确认是 NPLIB1 版本；最安全的做法是用 MIST 代码在比赛训练集上自己训练。**
- 现版本已自带 subformula 标注，不再依赖 SIRIUS。

**MIST-CF**（Goldman et al., 2023；[arXiv 2307.08240](https://arxiv.org/abs/2307.08240)；[代码](https://github.com/samgoldman97/mist-cf)）
- 从 MS/MS 推断前体分子式与加合物。README（已读）：公开权重只用 NPLIB1；NIST20 版只给持有 NIST 许可的人；需要 SIRIUS 5.5.7 的 `decomp` 做分子式枚举；只支持正离子模式。
- 对本赛：分子式推断收益很大（第 1 节）；但 SIRIUS 二进制要打包进离线 Notebook，且需核实其许可。替代方案是用 RDKit/自写的质量分解 + 碎片子式一致性打分。

**JESTR**（Kalia et al., Bioinformatics 2025；[arXiv 2411.14464](https://arxiv.org/abs/2411.14464)；[代码](https://github.com/HassounLab/JESTR1)）
- 已读原文：谱（1 Da 分箱 + MLP）和分子图（GCN）对比学习到同一空间，按余弦排序；训练末期加入同分子式 PubChem 候选作为负样本做正则。
- 结果（摘要，待核）：MassSpecGym 按质量 rank@1 15.13%（MIST 14.64%），按分子式 11.85%。NIST2020 上训练的版本存在。**若用其权重，只能用 MassSpecGym 或 NPLIB1 训练的版本。**
- 对本赛：候选感知的对比学习（把同质量/同分子式候选当负样本）是低成本、与你现有打分器兼容的训练目标。

**DreaMS**（Bushuiev et al., Nature Biotechnology 2025；[论文](https://www.nature.com/articles/s41587-025-02663-3)；[代码](https://github.com/pluskal-lab/DreaMS)）
- README（已读）：Transformer，在 GeMS（从 MassIVE GNPS 挖掘的数百万张无标注谱）上做掩码峰预测和保留时间顺序的自监督预训练；`dreams_embeddings()` 每张谱输出 1024 维；权重在 [Zenodo 10997887](https://zenodo.org/records/10997887) 和 Hugging Face；代码许可徽章为 MIT；需要 Python 3.11。README 未提 NIST。
- 风险："Separability Is Not Alignability" 指出发布的 DreaMS checkpoint 对 MassSpecGym 测试有污染，因此他们从 GeMS-A10 重训。对本赛这不是规则问题，但意味着论文里 DreaMS 在 MassSpecGym 上的数字偏高。
- timsTOF 适用性：GeMS 来自多种仪器，比只在 Orbitrap 上训练的模型更可能泛化到 timsTOF，但没有直接证据。

**SpecBridge**（Hassoun Lab，2026-01；[arXiv 2601.17204](https://arxiv.org/abs/2601.17204)；[代码](https://github.com/HassounLab/SpecBridge)）
- 已读原文：微调 DreaMS 最后两层，经残差映射 MLP 映射到冻结 ChemBERTa-2 的分子嵌入空间，用 MSE 对齐而非 InfoNCE。
- 结果（摘要，待核）：MassSpecGym R@1 84.7、MRR 88.1。**需结合 "in the Wild" 表 3 谨慎看待**：同一组合在修正候选规范化伪影后只有 6.70%。不要依据这个数字决定投入。

**Separability Is Not Alignability**（NeurIPS 2026 workshop；[arXiv 2610.37979](https://arxiv.org/abs/2610.37979)；[代码](https://github.com/autonlab/msms-retrieval)）
- 已读原文：在 MassSpecGym v1.5（MCES 划分）上，冻结 DreaMS 对比学习 MRR 0.227；微调 DreaMS 0.255；**用模拟谱预训练的全新编码器 + 候选感知 MaxSim 达到 MRR 0.566、R@1 0.425**。候选式 InfoNCE 会奖励“不看谱”的先验。
- 对本赛：这是我找到的检索方向最强、评估也较严谨的结果，且方法要素（模拟谱预训练）你已有原料。但算力（4×A100）超出 RTX 5070 一个数量级，需要缩小规模试。

### 3.3 谱图相似度

**Spectral entropy / Flash entropy search**（Li et al., Nature Methods 2021；Li & Fiehn, Nature Methods 2023；[代码 MSEntropy](https://github.com/YuanyueLi/MSEntropy)，Apache-2.0，`pip install ms_entropy`）
- 你已在用 entropy 检索。可补充的是：eganeganegan 报告把“原始强度 entropy 通道”的 12 个特征加入 LambdaRank 后公开分 0.185 → 0.226，说明 entropy 相似度不仅可作为检索器，也可作为 reranker 的多个特征（加权/不加权 entropy、匹配峰数、解释强度比例等）。

**MS2DeepScore 2 / MS2Query**（de Jonge et al.；[MS2Query, Nat Commun 2023](https://www.nature.com/articles/s41467-023-37446-4)；[跨离子模式 MS2DeepScore 预印本](https://www.biorxiv.org/content/10.1101/2024.03.25.586580)）
- 预测两张谱对应结构的 Tanimoto，用于类似物检索。环境线程已核实 MS2DeepScore 与主环境 pandas 冲突，需独立 venv；其预训练权重的训练数据（GNPS/MassBank 等公开库）需逐一确认不含 NIST。

### 3.4 正向谱模拟（结构 → 谱）

**ICEBERG**（Goldman et al., 2024；[arXiv 2304.13136](https://arxiv.org/abs/2304.13136)；代码 coleygroup/ms-pred）、**FraGNNet**（[arXiv 2404.02360](https://arxiv.org/abs/2404.02360)）、**MassFormer**（[arXiv 2111.04824](https://arxiv.org/abs/2111.04824)）、**GLACIER**（NeurIPS 2026，[arXiv 2606.29161](https://arxiv.org/abs/2606.29161)）
- 用途：为没有参考谱的 class 2 候选生成模拟谱，再做谱图匹配；或用模拟谱做编码器预训练/增强。
- MassSpecGym 摘要称 FraGNNet 是最强的谱模拟基线（待核）。
- 合规：你已在用 GLACIER/ICEBERG，权重来源审计见第 5 节（正在核实，结果会补入）。

### 3.5 de novo 生成（class 3）

| 方法 | 来源 | MassSpecGym top-1 / top-10 | 代码 | 备注 |
|---|---|---|---|---|
| DiffMS | ICML 2025，[arXiv 2502.09571](https://arxiv.org/abs/2502.09571) | 2.30% / 4.25%（摘要，待核） | [coleygroup/DiffMS](https://github.com/coleygroup/DiffMS) | 分子式约束图扩散；NPLIB1 8.34% |
| MBGen | AAAI 2026，[arXiv 2602.01643](https://arxiv.org/abs/2602.01643) | 7.58% / 12.54% | [biomed-AI/MBGen](https://github.com/biomed-AI/MBGen) | 多体注意力边扩散 |
| FRIGID | ICML 2026，[arXiv 2604.16648](https://arxiv.org/abs/2604.16648) | 9.05% / 12.23%（作者修订值；全 InChIKey 匹配下 4.97%）| [coleygroup/FRIGID](https://github.com/coleygroup/FRIGID) | SAFE 字符串掩码扩散 + ICEBERG 迭代精修；原报 18.29% 因 MIST 增强数据泄漏已撤回 |
| DualLGD | 2026-05，[arXiv 2605.07048](https://arxiv.org/abs/2605.07048) | 23.89% / 33.08%（作者自报） | [du-lab-data-science/DualLGD](https://github.com/du-lab-data-science/DualLGD) | 线图双流扩散；需 4×H200 预训练（摘要） |
| GEMS | 2026-10，[arXiv 2610.23617](https://arxiv.org/abs/2610.23617) | Hit@1 30.22% / Hit@10 41.33%（协议不同：从 PubChem 同分异构体出发搜索） | 未发布 | 已知分子式下在同分异构体图上做学习型搜索 |

对本赛的判断：
- 上述数字都假设已知分子式，且以 InChIKey 精确命中计。比赛以 InChIKey 前 14 位计分，与 MassSpecGym 的 2D 匹配口径接近。
- 即便按最乐观的 DualLGD 23.89% top-1，class 3 在隐藏集中所占比例未知，且 timsTOF 与训练分布不同；推理成本（DiffMS 约 131 秒/谱，FRIGID-base 约 6.6 秒/谱，单张 RTX 6000 Ada）对 1,500 张谱、9 小时限制来说 FRIGID 量级尚可，DiffMS 量级不可行。
- 实用折中：只在检索置信度低时，把生成候选放在第 k 位之后（不挤掉高置信检索结果），MRR@25 的尾部位置仍有价值。
- "in the Wild" 指出 FRIGID 与 MIST+MolForge 原报数字都因泄漏或 bug 大幅下修（MIST+MolForge top-1 31.75% → 3.29%，摘要待核），**对 de novo 方向的论文数字要打折看**。

---

## 4. 历届 CASMI 的经验

| 年份 | 设计 | 优胜方法与结果 | 来源 |
|---|---|---|---|
| 2013 | 4 个类别 | 类别 1 R. Dunn 团队 14 题中 8 题第一；类别 2 H. Oberacher 5 题中 3 题正确 | [casmi-contest.org 2013 结果页](http://www.casmi-contest.org/results.shtml)；综述 [Metabolites 2013](https://doi.org/10.3390/metabo3020412) |
| 2016 | 208 题（Q Exactive，正 127/负 81），ChemSpider ±5 ppm 候选 | 类别 2（仅碎片计算）：CSI:IOKR 赢 41% 题目；类别 3（可用元数据）：MS-FINDER + 谱库检索赢 76%，Top1 约 70% | [Schymanski et al., J Cheminform 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)（摘要部分已读原文） |
| 2017 | 4 个类别 | SIRIUS/CSI:FingerID 赢类别 1、2、4；类别 4 正确 66/198（33.3%），是最佳非 CSI 参赛者的 6 倍以上；数据无同位素模式，分子式常被排错 | [Böcker 组博客](https://bio.informatik.uni-jena.de/2017/11/casmi-2017-results-are-out) |
| 2022 | 500 个化合物（优先 250），正负模式原始 LC-MS/MS，不在公开谱库中；分别评加合物、分子式、化合物类别、2D 结构 | 获奖名单和方法未能从官方页读到（结果放在 Google Drive），未找到正式综述论文 | [Fiehn 实验室 CASMI 2022 页](https://fiehnlab.ucdavis.edu/casmi) |

可迁移到本赛的经验（CASMI 2016 部分来自摘要，待核）：
1. **元数据先验非常强**：2016 类别 3 只用参考文献计数就 Top1 143/208。对应本赛的 PubChem/文献热度先验。但 2016 的题目多为常见人为化学品，先验被高估；本赛是天然产物且含新分子，先验应作为 reranker 特征而不是硬规则。
2. **等权集成稳健**：6 个碎片打分器等权组合 Top1 82，高于最佳单模型 70。这与你的 RRF 思路一致；建议在 reranker 之外保留一个等权/秩平均的对照，防止 reranker 过拟合验证分布。
3. **训练集重叠会虚高**：2016 年 CSI:FingerID 在 67 题中排第一的候选就在其训练集里；对缺席训练集的题目做留出测试，Top1 从 34% 降到 18%。对应本赛：本地验证必须结构不相交。
4. **分子式错误是主要失败来源**（2017）：没有同位素信息时分子式容易排错。本赛 timsTOF 隐藏集的元数据有哪些（是否有 MS1 同位素）应在 test.parquet 的 12 列中确认。
5. **谱图划分本身会泄漏**（2016）：用 MassBank 相似度 > 0.85 划分训练/挑战题后，再把 MoNA 相似度当特征反而让成绩变差。对应本赛：如果用谱相似度挑验证集，不要再用同一相似度给验证集打分来调权重。

---

## 5. 外部数据与预训练模型合规核查

规则依据（环境线程 2026-11-24 已从 Kaggle 官方页读取，本线程直接访问 Kaggle 被权限策略拦截，未绕过）：允许免费公开的外部数据和预训练模型；主办方讨论帖 #750330（2026-10-17）澄清允许 CC BY-NC 等非商业开源数据；**NIST 明确禁止，包括其衍生模型权重**；非开源、付费或专有数据违规。关于“预训练权重是否须公开训练代码和数据”，截至环境线程读取时主办方未答复；关于 CC BY-NC checkpoint 是否合格另有讨论帖 #746430（FRIGID issue #10 引用），本线程未能读取。

| 资源 | 许可 | NIST 风险 | 结论 |
|---|---|---|---|
| COCONUT | CC BY 4.0（仓库附署名）；christianmcb 用的缓存标 CC BY-NC-SA | 无 | 可用 |
| PubChem / PubChemLite | 公共领域 / CC BY 4.0 | 无 | 可用；热度先验可用 |
| LOTUS | CC BY 4.0 | 无 | 可用 |
| NPAtlas | CC BY-NC 4.0 | 无 | 按 #750330 允许 |
| ChEMBL | CC BY-SA 3.0 | 无 | 可用 |
| MassSpecGym / MSnLib / GNPS / MassBank / MoNA | 公开（具体许可待逐一核实） | MassSpecGym 不含 NIST | 可用；注意与比赛训练集重叠时的验证泄漏 |
| DreaMS 权重 | 代码 MIT（README 徽章），权重许可未核实 | README 未提 NIST；预训练数据来自 GNPS/MassIVE | 大概率可用，需看 Zenodo 记录许可 |
| MIST 官方权重 | 未见许可声明 | **主基准 csi2022 含 NIST**，下载的权重来源未说明 | **不建议用**；用 MIST 代码在比赛数据上自训 |
| MIST-CF 公开权重 | 未见许可声明 | 公开版只用 NPLIB1；NIST 版不公开 | 可用公开版；依赖 SIRIUS decomp |
| JESTR 权重 | 未核实 | 论文有 NIST2020 训练版本 | 只用 MassSpecGym/NPLIB1 版本，或自训 |
| ICEBERG（ms-pred） | MIT（FRIGID issue 转述） | **NIST 训练版另行门控发放** | 必须确认你用的是 NPLIB1/MassSpecGym 版本（核实中） |
| GLACIER | 待核 | 待核 | 核实中 |
| FRIGID checkpoint | Zenodo 标 CC BY-NC 4.0，代码 Apache-2.0，issue #10 无维护者回复 | 其 MIST 编码器用 GLACIER 模拟谱增强，链式依赖 GLACIER | 待 GLACIER 结论 |
| FPNet（Kaggle 用户发布） | 作者称沿用比赛数据条款 | 未说明 | 如使用，需向发布者确认训练数据 |
| CFM-ID | — | 2016 版 CFM_retrain 用了 METLIN + NIST | 避免使用 NIST 训练的 CFM-ID 模型 |
| SIRIUS / CSI:FingerID | — | CSI:FingerID 指纹预测依赖在线 Web 服务 | 比赛 Notebook 必须关网，**CSI:FingerID 不可用**；仅离线 `decomp` 可考虑 |

---

## 6. 建议的下一步实验（交给实验线程排期）

1. **合规审计**（0.5 天）：列出 0.495 Notebook 挂载的所有 Kaggle 数据集与权重，逐个记录训练数据来源与许可；任何含 NIST 的权重立即替换。
2. **候选规范化与盲谱检查**（0.5 天）：统一 RDKit 规范化所有候选 SMILES；用常数谱跑一次全流程，看排序器在无谱信息时的 MRR，量化非谱先验与潜在捷径。
3. **打分器训练目标消融**（1–2 天 GPU）：同一编码器，按位 BCE vs 候选内 listwise（同质量硬负样本，参考 FPNet/JESTR）。结构不相交验证。
4. **分子式一致性特征**（1–2 天）：对每个候选的分子式计算可解释峰比例、解释强度、子式树得分，加入 reranker；消融看 class 2 代理上的增益。
5. **热度先验**（0.5 天）：若尚未使用，加入 ln(1+PubChem 记录数) + ln(1+PubMed 数) 作为 reranker 特征，单独报告对已知/未知代理的影响。
6. **DreaMS 嵌入作为额外特征**（1–2 天）：先离线算训练集与候选模拟谱的嵌入，测 CPU 推理时间是否能放进 9 小时。
7. 仅当以上完成且有余量：模拟谱预训练编码器（小规模）或 FRIGID 尾部补充生成候选。

每项都需要在新的、未观察过的结构不相交留出集上验收，公开榜只作最终外部反馈。

---

## 7. 未完成与需你确认的事项

- 没有读到你的 0.495 Notebook 代码，第 1 节“你可能缺的部分”都是推测。
- Kaggle 讨论帖 #746430（CC BY-NC checkpoint 是否合格）和 #750330 的后续回复，本线程未能读取，需要你在浏览器中查看。
- CASMI 2022 的获奖方法、MassSpecGym 与 JESTR 结果表、CASMI 2016 详细表格，都来自截断页面的摘要，引用前需对照 PDF。
- GLACIER 与 ICEBERG 公开权重的训练数据正在核实，结论会补入第 5 节。
