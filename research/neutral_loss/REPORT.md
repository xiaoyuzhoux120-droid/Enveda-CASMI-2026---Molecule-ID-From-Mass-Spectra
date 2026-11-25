# 研究方向 2：中性损失、诊断碎片、官能团识别与化学先验

日期：2026-11-24 · 运行环境：云端 4 CPU，无 GPU · 全部实验在本地 train.parquet 上完成，**没有提交 Kaggle**，没有使用测试集标签或付费服务。

## 1. 结论摘要

1. **当前最佳管线（Kaggle notebook `casmi26-nightly-frp5-20261119`，public 0.495）几乎没有用到中性损失先验。** 其中唯一的“中性损失 ↔ 官能团” 组件是 RR8 LightGBM 异构体 reranker（13 个手写损失的 agree/clash 特征），但配置 `V8 = 'MASS'` 使 `RR8 = False`，即处于关闭状态。真正在线的化学/物理检查是 MetFrag-lite（1–2 键断裂解释强度）。
2. **验证集陷阱：** train 中 `enveda-180`（115 万谱，约 18.3 万结构）是合成类药化合物库（Enamine 风格酰胺/磺酰胺），不是天然产物；而测试集是天然产物。按 train 随机抽样构造同分子式异构体面板时，98.7% 的查询来自 enveda-180。中性损失词表若在全体数据上挖掘，也会被该库的组合化学片段（如 129.079、182.106 Da）主导。本研究所有词表挖掘与评测都排除了 enveda-180。
3. **数据驱动的高分辨中性损失与官能团强相关，且能在留出分子上复现。** 在 400 个挖掘出的损失中心中，有 426 条“损失 → 子结构” 关联满足 lift ≥ 3；在完全不重叠的天然产物留出分子上，98.1% 仍保持 lift > 1.5。典型例子：162.053 Da（己糖）→ 己糖环 lift 16.3（留出 13.3），146.058（脱氧己糖）lift 15.8（留出 15.4），56.063（C4H8）→ 异戊烯基 lift 9.0（留出 11.0）。
4. **同分子式异构体排序（MRR，面板中位 8 个候选）：** MetFrag-lite 0.712 远强于任何仅靠谱图学习的先验（最佳 0.506）；但在学习型先验内部，**高分辨损失/碎片词表比整数 bin 提升 +0.062**（95% CI [0.051, 0.073]），**以官能团 SMARTS + MACCS 为目标比 Morgan 2048 提升 +0.077**（CI [0.061, 0.092]），两项在官方 np-examples 上同样显著。
5. **学习型先验与 MetFrag-lite 线性融合没有增益**（在半 A 选出的权重为 0，在半 B 和 np-examples 上差值 CI 均跨 0）。
6. **机制性损失项放进 GBDT reranker 有小而稳定的增益：** 同一训练面板、同一模型，仅改变损失特征。13 个 RR8 损失 +0.008（CI [0.001, 0.015]），扩展的 31 个机制性损失 +0.012（CI [0.005, 0.020]），均相对仅用 MetFrag 特征的 GBDT，在 2,000 个天然产物异构体面板上测得。在 207 个 np-examples 面板上方向一致但 CI 跨 0；相对原始 MetFrag-lite 分数，31 损失 GBDT 为 +0.037（CI [0.002, 0.073]）。

**建议：** 在 0.495 管线里重新启用 RR8 类异构体 reranker，但 (a) 用天然产物库的面板训练，不要用 enveda-180；(b) 把损失词典从 13 个扩展为本文的 31 个；(c) 保留 RR8 的“库内强匹配不动、训练集结构冻结” 护栏。因为它只重排同分子式候选，对公开分数的影响会被稀释，必须先在 Kaggle 上作为单独版本验证。

## 2. 现有方案中的化学先验盘点

| 位置 | 内容 | 状态 |
|---|---|---|
| repo `casmi_ml/data.py` | 中性损失 = precursor − m/z，按 **1 Da 整数 bin** 做 1250 维直方图；峰 token 也带损失的正余弦编码 | 用于 repo 中的 MLP/Transformer 指纹模型（公开分 ≤0.176，已过时） |
| notebook `frag_rescore`（MetFrag-lite） | 枚举 ≤2 键断裂碎片，H 位移 −2..+3，0.005 Da，按解释强度打分 | **启用**（`PHYSICAL = True`） |
| notebook `rr_features` + `rr_overlay`（RR8） | 13 个损失（H2O, CO2, HCOOH, NH3, CH3, CH3OH, C2H2O, 己糖, 脱氧己糖, 戊糖, 葡糖醛酸, SO3, CO）× SMARTS 的 agree/clash + MetFrag 特征，LightGBM | **关闭**（`V8 = 'MASS'`） |
| notebook 其他 | 质量误差先验、PubChem popularity、NP knowledge table、文库命中门控 | 启用；与中性损失无关 |

## 3. 数据与验证设计

脚本：`research/neutral_loss/prep.py`。

- 只用 `[M+H]+` 与 `[M-H]-`（测试集 1213 条谱中 1152 条属于这两类）。
- 按 `sha256('nl:' + inchikey14)` 分桶，分子级不重叠：bucket < 8500 为 **fit**（56,833 分子 / 159,044 谱，每分子 ≤3 谱）；bucket ≥ 9500 为 **eval**。
- **eval**：2,000 个分子，要求 (1) 至少有一条非 enveda-180 谱；(2) catalog 中存在同分子式的其他结构。只取非 enveda-180 的谱（4,686 条）。
- **np**：官方 `enveda-np-examples` 的全部 845 条 `[M+H]+/[M-H]-` 谱（234 个结构，其中 207 个有同分子式异构体），这些结构已从 fit 中剔除。
- **异构体面板**：真值 + 从 275,810 个 catalog 结构中随机抽取的同分子式其他结构，最多 40 个，中位 8 个。这模拟的是“分子式已正确”之后最难的一步：同分子式异构体排序。
- **指标**：真值在面板中的 reciprocal rank；同分时按随机打破同分取期望值，避免常数分数“白拿” 名次。配对 bootstrap 2,000 次给出 95% CI。

局限：面板中的诱饵来自训练库结构，不等同于比赛实际的 COCONUT/PubChem/LOTUS 候选池；面板大小（中位 8）也小于真实候选池。所以这里的 MRR 不能直接换算成 Kaggle 分数，只适合比较方法之间的相对差异。

## 4. 实验 A：高分辨中性损失挖掘与官能团关联

脚本：`research/neutral_loss/mine.py`，`research/neutral_loss/chem.py`。

- 在 fit 中非 enveda-180 的 `[M+H]+` 谱上，以 1 mDa 分辨率统计“分子级” 损失出现直方图（相对强度 ≥0.5%，去掉前体离子附近），平滑后选峰，得到 400 个损失中心（2–500 Da）与 400 个碎片离子中心（40–500 m/z），负离子模式另有 150 个损失中心。结果在 `results/vocab.json`。
- 损失与 42 个面向天然产物的 SMARTS 官能团（`chem.NP_SMARTS`）在分子级做 lift = P(官能团 | 观察到该损失，强度 ≥2%) / P(官能团)。

最常见损失（占 fit 分子比例）：H2O 32.7%、HCOOH（H2O+CO）16.7%、NH3 9.6%、C2H4O2 8.7%、2H2O 8.0%、CO 6.0%、CH3OH 5.8%、CO2 5.0%、C4H8 4.9%、CH3• 4.7%、C2H2O 4.4%。

机制明确的关联（fit = 训练分子，eval = 不重叠的天然产物留出分子；eval lift 已做加一平滑）：

| 损失 (Da) | 解释 | 官能团 | fit 中出现该损失的分子数 | P(官能团) | P(官能团 \| 损失) fit | lift fit | eval 分子数 | P(官能团 \| 损失) eval | lift eval |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 162.0531 | C6H10O5 己糖 | 己糖环 | 290 | 0.029 | 0.469 | 16.3 | 49 | 0.510 | 13.3 |
| 146.0585 | C6H10O4 脱氧己糖 | 脱氧己糖环 | 222 | 0.008 | 0.122 | 15.8 | 51 | 0.216 | 15.4 |
| 180.0638 | C6H12O6 | 己糖环 | 254 | 0.029 | 0.406 | 14.1 | 42 | 0.333 | 8.7 |
| 15.0237 | CH3• 自由基 | flavone 骨架 | 453 | 0.011 | 0.102 | 9.5 | 88 | 0.193 | 6.7 |
| 56.0625 | C4H8 | prenyl | 517 | 0.010 | 0.093 | 9.0 | 71 | 0.197 | 11.0 |
| 36.0211 | 2H2O | 甾体骨架 | 842 | 0.032 | 0.248 | 7.8 | 170 | 0.253 | 5.6 |
| 42.0107 | C2H2O 乙烯酮 | 乙酰酯 | 402 | 0.018 | 0.139 | 7.6 | 80 | 0.125 | 5.5 |
| 131.0724 | — | 吲哚 | 233 | 0.040 | 0.300 | 7.5 | 42 | 0.167 | 3.9 |
| 60.0211 | C2H4O2 乙酸 | 乙酰酯 | 856 | 0.018 | 0.109 | 5.9 | 170 | 0.165 | 7.3 |
| 17.0266 | NH3 | 伯胺 | 948 | 0.088 | 0.430 | 4.9 | 151 | 0.530 | 5.1 |
| 32.0265 | CH3OH | 甲氧基 | 607 | 0.183 | 0.792 | 4.3 | 91 | 0.813 | 3.7 |
| 46.0054 | HCOOH | 羧酸 | 1755 | 0.230 | 0.793 | 3.4 | 327 | 0.743 | 2.5 |
| 27.9951 | CO | 香豆素 | 592 | 0.016 | 0.047 | 2.9 | 102 | 0.108 | 4.9 |

整体：426 条强关联（fit 中损失分子数 ≥50、共现 ≥20、lift ≥3、eval 中损失分子数 ≥20），98.1% 在 eval 上 lift > 1.5；log-lift 的 fit/eval 相关系数 0.53。

注意：很多大质量损失（例如 100–250 Da）与 `lactone`、`enone` 的高 lift 来自化合物家族共现（同一骨架系列），并非可泛化的断裂机理。所以实验 C 的扩展词典只收录有明确化学机理的损失。完整表格：`results/lift_strong.csv`。

## 5. 实验 B：学习型子结构先验 vs MetFrag-lite（同分子式异构体排序）

脚本：`research/neutral_loss/rank_exp.py`，`research/neutral_loss/analyze.py`。

学习型先验是一个 2 层 MLP（512 隐层，8 epoch，CPU 每个约 1 分钟），在 fit 上训练，输出子结构概率。候选的得分是其子结构向量在预测概率下的 Bernoulli 对数似然（CSI:FingerID 风格打分）。

- 特征 `int`：1 Da 整数 m/z 直方图 + 1 Da 整数损失直方图（即 repo `casmi_ml` 的表示）。
- 特征 `hr`：`int` + 挖掘出的 400 损失 + 400 碎片高分辨词表（±4 mDa，sqrt 强度）。
- 目标 `chem`：42 个 NP SMARTS + 166 MACCS；目标 `morgan`：Morgan r2 2048。

| 方法 | eval MRR (2000) | eval Top-1 | np-examples MRR (207) | np Top-1 |
|---|---:|---:|---:|---:|
| 随机 | 0.379 | 0.185 | 0.368 | 0.171 |
| RR8 13 损失 agree−clash（无训练） | 0.431 | 0.233 | 0.423 | 0.225 |
| nn_int_morgan | 0.400 | 0.205 | 0.361 | 0.150 |
| nn_hr_morgan | 0.429 | 0.231 | 0.380 | 0.179 |
| nn_int_chem | 0.444 | 0.252 | 0.430 | 0.242 |
| **nn_hr_chem** | **0.506** | **0.314** | **0.487** | **0.305** |
| **MetFrag-lite**（notebook DEFAULT_CFG） | **0.712** | **0.568** | **0.587** | **0.408** |

配对消融（MRR 差值，95% CI）：

| 对比 | eval | np-examples |
|---|---|---|
| hr vs int（chem 目标） | +0.062 [0.051, 0.073] | +0.057 [0.028, 0.085] |
| hr vs int（Morgan 目标） | +0.029 [0.022, 0.036] | +0.019 [0.002, 0.037] |
| chem vs Morgan（hr 特征） | +0.077 [0.061, 0.092] | +0.107 [0.062, 0.153] |

随面板变大，MetFrag-lite 的优势增大（26–40 个候选时 MRR 0.669 vs nn_hr_chem 0.341）。

融合：`z(MetFrag) + w·z(prior)`，w 只在 eval 半 A 上选择。四个 NN 先验选出的 w 都是 0。手动检查 w = 0.02–0.2，在半 B 与 np-examples 上的差值 CI 均跨 0，np-examples 上点估计为负。RR8 的 13 损失项 w = 0.05 时，半 B 为 +0.0027 [0.0007, 0.0049]，np 为 +0.004 [0.0001, 0.0099]，增益很小但稳定。

**解读：** 统计型子结构先验学到的主要是“这个谱像哪类化合物”。同分子式异构体往往同类，所以这部分信息和 MetFrag-lite 的断裂解释高度重叠。机制型 agree/clash（“观察到 −162.053，候选必须有己糖”）才提供与 MetFrag 互补的约束。

## 6. 实验 C：GBDT 异构体 reranker 的损失特征消融

脚本：`research/neutral_loss/combiner.py`。

- 训练面板：fit 中 4,000 个有同分子式异构体的天然产物类分子（只用其非 enveda-180 谱）。测试面板：上述 eval 2,000 + np 207，与训练分子不重叠，测试集上不调任何参数。
- 模型：sklearn `HistGradientBoostingClassifier`（300 轮，lr 0.05），pointwise，特征同时给原值与面板内去均值值，与 RR8 的 `centre_within` 一致。
- 三个臂只差损失特征：`mf_only`（MetFrag DEFAULT + BASELINE 两种配置 + 5 个候选描述符）、`rr8_13`（+13 损失 agree/clash）、`ext_31`（+31 损失，见 `combiner.EXT_LOSSES`：新增 2H2O（需 ≥2 个 OH）、CH2O、C2H4O2、C4H8、C5H8、HCN、CH5N、C2H7N、C3H9N、己糖+H2O、脱氧己糖+H2O、香豆酰、咖啡酰、阿魏酰、没食子酰、丙二酰、H2S、HCl）。

| 臂 | eval MRR | np MRR |
|---|---:|---:|
| 原始 MetFrag-lite 分数 | 0.7134 | 0.5936 |
| GBDT mf_only | 0.7086 | 0.6227 |
| GBDT rr8_13 | 0.7167 | 0.6272 |
| **GBDT ext_31** | **0.7207** | **0.6302** |

| 配对差值（95% CI） | eval | np |
|---|---|---|
| rr8_13 − mf_only | +0.0081 [0.0011, 0.0149] | +0.0045 [−0.0216, 0.0325] |
| ext_31 − mf_only | **+0.0121 [0.0048, 0.0195]** | +0.0075 [−0.0209, 0.0357] |
| ext_31 − rr8_13 | +0.0040 [−0.0012, 0.0091] | +0.0030 [−0.0167, 0.0206] |
| ext_31 − 原始 MetFrag | +0.0073 [−0.0025, 0.0170] | **+0.0366 [0.0022, 0.0734]** |

两次运行结果完全一致（固定种子）。结论：损失特征的贡献在 2,000 面板上显著；扩展词典相对 13 个手写损失的额外收益为正，但尚未达到统计显著。

## 7. 对 0.495 管线的建议（按优先级）

| 优先级 | 动作 | 证据 | 成本 | 风险 |
|---|---|---|---|---|
| P0 | 用天然产物库面板（排除 enveda-180）训练 RR8 类 reranker，特征使用 `ext_31`，保留 `lib_tau` 门控和训练集结构冻结，作为**单独 notebook 版本**在 Kaggle 上 A/B | 实验 C：+0.012 MRR（异构体子问题） | CPU：4,000 面板特征约 5 分钟；推理每分子毫秒级 | 只影响同分子式候选；公开分数可能变化不明显；RR8 原训练数据来源不明，可能含 enveda-180 偏差 |
| P1 | 神经打分器的损失输入从 1 Da bin 改为高分辨词表（或 MIST 式 subformula 编码） | 实验 B：+0.03 到 +0.06 | 只改特征，重训一次 | 需在 notebook 的双神经打分器上复现 |
| P1 | 给指纹/嵌入模型加官能团（SMARTS + MACCS）辅助头，多任务训练 | 实验 B：chem 目标对异构体区分 +0.08 到 +0.11 | 很低 | 主任务指标可能受影响，需消融 |
| P2 | 把挖掘出的 lift 关联做成可解释的 naive-Bayes 先验，只用于候选池扩展时（COCONUT/PubChem 未知结构）的预过滤 | 实验 A | 低 | 家族共现型关联会带来偏差，只能使用机制明确的关联 |
| P2 | 负离子模式单独的损失词典（CO2、SO3、C6H8O6 葡糖醛酸等），本实验只挖掘了 150 个 | 测试集 226 条负离子谱 | 低 | 样本少 |

**不建议**：把学习型子结构先验以固定权重线性加到 MetFrag 上（实验 B 无增益，np 上点估计为负）。

## 8. 文献与公开实现

| 工作 | 与本方向的关系 | 链接 |
|---|---|---|
| MIST（Goldman et al., *Nat. Mach. Intell.* 5, 965–979, 2023） | 给峰做 subformula 注释，用 formula transformer 隐式编码成对中性损失，并预测子结构片段作为辅助任务 | https://www.nature.com/articles/s42256-023-00708-3 · https://github.com/samgoldman97/mist |
| MassSpecGym（Bushuiev et al., arXiv 2410.23326） | 标准检索基准，MCES 划分；MIST hit@1 14.64% | https://arxiv.org/abs/2410.23326 |
| MassSpecGym in the Wild（Liu et al., arXiv 2606.19624，含 Enveda 作者） | 已知分子式时，随机 hit@1 从 0.43% 升到 12.98%；MIST 4M-aug 40.31%、ICEBERG 36.18%、PubChem 默认排序 49.98%；指出 SMILES 规范化捷径会虚高分数 | https://arxiv.org/abs/2606.19624 |
| Nearest-neighbour baselines for fingerprint prediction（Khoo, arXiv 2610.02249） | subformula 规则 + 中性损失 token，在 NPLIB1 scaffold 划分上 Jaccard +0.078；neutral-loss cosine 不优于普通 cosine | https://arxiv.org/abs/2610.02249 |
| Bittremieux et al., *JASMS* 33, 1733 (2022) | 1,000 万对小分子谱比较：modified cosine 在所有情形下优于 neutral-loss matching 和 cosine | https://doi.org/10.1021/jasms.2c00153 |
| Spec2Vec（Huber et al., *PLoS Comput Biol* 2021） | 把 `loss@xxx.xx`（5–200 Da）作为与 `peak@` 并列的词嵌入 | https://doi.org/10.1371/journal.pcbi.1008724 |
| MS2LDA（van der Hooft et al., *PNAS* 2016） | 把碎片与损失共现分解成 Mass2Motif 子结构基序（原文页面被验证码拦截，未逐字核对） | https://doi.org/10.1073/pnas.1608041113 |
| CANOPUS（Dührkop et al., *Nat. Biotechnol.* 39, 462, 2021） | 从谱预测 2,497 个 ClassyFire 化合物类别，交叉验证平均准确率 99.7%；化合物类别先验 | https://www.nature.com/articles/s41587-020-0740-8 |
| ICEBERG（Goldman, Li, Coley, arXiv 2304.13136） | 生成碎片 DAG 预测谱；在天然产物集上 top-1 检索 29% vs 20% | https://arxiv.org/abs/2304.13136 |
| FraGNNet（Young et al., arXiv 2404.02360） | 确定性碎片枚举 + 概率模型，用于谱预测检索 | https://arxiv.org/abs/2404.02360 |

后续值得深入：(1) MIST 式 subformula 注释，需要先解决分子式；比赛候选已带分子式，可以对每个候选单独做“候选条件下的注释”；(2) CANOPUS 式类别先验对候选池扩展的作用；(3) MS2LDA 基序作为 GBDT 特征；(4) 碎片–碎片成对差值（不仅是 precursor − fragment），即 MIST 的隐式成对损失。

## 9. 复现

```bash
.venv/bin/kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data && (cd data && unzip -o *.zip)
# MetFrag-lite：从 notebook casmi26-nightly-frp5-20261119 的 `_physical` 字符串中导出为 frag_rescore.py，放到 $W
W=/path/to/work
.venv/bin/python research/neutral_loss/prep.py data $W 60000 2000      # ~1 min
.venv/bin/python research/neutral_loss/mine.py $W                      # ~2 min
.venv/bin/python -u research/neutral_loss/rank_exp.py $W $W 40         # ~6 min（CPU）
.venv/bin/python research/neutral_loss/analyze.py $W
.venv/bin/python -u research/neutral_loss/combiner.py $W $W 4000       # ~8 min
```

`results/` 中保存了 `summary.json`、`combiner_summary.json`、`vocab.json`、`lift_strong.csv`。`frag_rescore.py` 来自 Kaggle 公开 notebook 代码，没有提交到本仓库。

## 10. 追加：在 FRP5 的 RR8 特征上重训 lambdarank（2026-11-24，Kaggle A/B 前的本地门槛）

脚本：`research/neutral_loss/rr8_retrain.py`。直接复用 notebook 的 `rr_features`（102 维：MetFrag 风格解释度、峰覆盖、候选描述符、13 损失 agree/clash）和 `rr_overlay.model_features`。31 损失版本（`rr_features_ext`，170+4 维）只替换损失词典。在 8,000 个天然产物 fit 面板上训练 LightGBM lambdarank（200 树，15 叶，lr 0.05）。测试面板与第 6 节相同，测试集上不调任何参数。

| 排序器 | eval MRR (2000) | np MRR (207) |
|---|---:|---:|
| MetFrag-lite | 0.7124 | 0.5883 |
| notebook 原版 RR8 booster（13 损失） | 0.7451 | 0.6190 |
| **重训 13 损失** | **0.7585** | **0.6510** |
| 重训 31 损失 | 0.7546 | 0.6455 |

| 配对差值（95% CI） | eval | np |
|---|---|---|
| 重训 13 − 原版 | +0.0133 [0.0050, 0.0219] | +0.0320 [0.0046, 0.0607] |
| 重训 31 − 重训 13 | −0.0039 [−0.0096, 0.0016] | −0.0055 [−0.0280, 0.0156] |
| 重训 31 − MetFrag | +0.0422 [0.0311, 0.0528] | +0.0572 [0.0200, 0.0954] |

结论更新：在 RR8 已有的丰富谱图特征上，31 损失扩展**没有**进一步提升（与第 6 节的简化特征 GBDT 结论不同）。主要收益来自用天然产物面板重训。因此 Kaggle A/B 上线的是“重训 13 损失 booster + `V8 = 'BOTH'`”，特征代码与原 notebook 逐字节一致，只替换模型字符串（sha256 `6ad35fa1…e6cf`）。原版 booster 的训练数据未知，可能与本评测分子重叠，这只会让原版的数字偏高。

### 10.1 Kaggle 结果（2026-11-25）

提交 58847206（notebook `xiaoyuzhoux120/casmi26-frp5-rr8np-20261124` v1，FRP5 + `V8='BOTH'` + 重训 RR8）公开分 **0.486**，低于 FRP5 对照的 **0.495**（−0.009，约 2 个分子的差距）。公开榜约 125 个分子，单个分子约 0.004，这个差距超出了噪声范围，但还不足以下定论。结论：本地同分子式异构体面板上的提升没有转化为比赛收益；打开 RR8 不应进入主线，0.495 仍是最佳。可能原因：(1) 本地面板的诱饵是训练库结构，而线上 RR8 只重排非训练库候选（COCONUT/PubChem/LOTUS），分布不同；(2) 线上融合使用 `z(-log 位置) + z(模型)`，会打乱上游多级 reranker 已经排好的顺序；(3) 线上同分子式组内候选数和本地不同。下一步如果继续，应在本地复现“上游排序 + RR8 融合”的完整设定，而不是单独评估 booster。
