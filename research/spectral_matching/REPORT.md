# 研究方向 1：m/z 对齐、mass tolerance 与谱图匹配算法

日期：2026-11-24 · 范围：CASMI 2026 实际最佳管线（Kaggle notebook `xiaoyuzhoux120/casmi26-nightly-frp5-20261119`，public MRR@25 **0.495**）中的谱图匹配环节。全部数字来自本目录脚本在 `train.parquet` 上的实际运行（CPU，无 GPU），未做任何 Kaggle 提交。

## 0. 结论摘要

| # | 发现 | 证据 | 建议 |
|---|---|---|---|
| 1 | 管线的 **analog search**（±200 Da）用 `max(unshifted, shifted)` 两次独立 entropy，且保留前体峰、0.01 Da 固定 tolerance。改为 **一次性 one-to-one modified (hybrid) entropy + 去前体峰 + 0.002 Da + 10 ppm** 后，top-1 analog 的 Tanimoto 0.314 → **0.382**（+0.068，CI95 [+0.061, +0.075]） | Task B，2,000 个留出分子 | **最高优先级**，但需重算 ranker 训练特征（见第 6 节） |
| 2 | 同样的改动使"用 analog 证据给同分异构体候选排序"的 MRR 0.593 → **0.716**（+0.124，CI95 [+0.111, +0.136]） | Task C，1,926 个分子，中位 39 个同式异构体候选 | 可作为无训练的后置层（类似 neighbour network）先上 |
| 3 | **extlib rank-1 overlay** 用 sqrt-cosine（0.015 Da，TAU 0.80）。改用 weighted entropy、TAU 0.65–0.70 在三个指标上同时更好：正确提升 0.868 → 0.937/0.909，错误提升 0.012 → 0.006/0.005，"新分子"误提升 0.046 → 0.040/0.029 | overlay 校准，2,225/2,250 分子 | 中优先级，改动小，只影响 rank 1 |
| 4 | timsTOF 碎片 m/z 实测误差：中心 **+2.5 ppm**，95% 在约 ±10 ppm 内，绝对误差 q95 ≤ 0.003 Da（m/z < 600）。0.01 Da 在 m/z 100 处相当于 100 ppm，过宽；且 timsTOF 谱中常见相距 0.003 Da 的相邻峰 | 400 分子、24,449 峰的 sub-formula 注释 | 统一使用 `0.002 Da + 10 ppm` |
| 5 | **身份检索**（库里有同一结构）已接近饱和（MRR 0.988），tolerance、峰处理的差异都 < 0.002；weighted entropy 明显优于 cosine（+0.005~0.023） | Task A | 身份检索通道保持现状即可 |
| 6 | **neighbour network** 的 modified cosine（0.0101 Da，0.01 Da 取整）在 τ=0.6 下每个查询有约 42 条边、精度（Tanimoto≥0.6）仅 0.8%；改 10 ppm tolerance 后边数降到 27、精度 1.1%；modified entropy 的 AUC 最高（0.939 vs 0.917） | 600 分子、1,970 万对 | 低优先级；网络层的 τ 需按新分数重调 |
| 7 | Neutral-loss 谱融合、峰数上限（64/128/256）、0.005 Da centroid、min m/z 50 在 analog 任务上均无显著收益 | Task B | 不建议继续投入 |

一句话：**瓶颈不在"同一化合物能否匹配上"，而在"新分子的结构类似物找得准不准"**。隐藏测试集多为库外天然产物，analog 证据是管线的关键输入，而这里有一个便宜、可复现、幅度大的改进。

## 1. 管线中的四处谱图匹配（审查结果）

从 FRP5 notebook 解出内嵌的 frozen 0.465 管线（`FROZEN_B64`，SHA-256 `d1e757c2…`）及 `ENG_FILES['pv.py' / 'casmi_engine.py']`：

| 位置 | 算法 | 参数 |
|---|---|---|
| 库检索 `molecule_channels` → `pv.search` / `search_shift_rows` | weighted spectral entropy（Li et al. 2021），顺序 merge 匹配 | 0.01 Da；相对强度下限 0.002；top 256 峰；前体峰保留（`mz <= prec + 2`）；10 ppm 前体窗 |
| analog 检索 `analog_search` → `pv.search_shift_pre` | 同上，`max(unshifted, shifted)`，shift = 中性质量差 | ±200 Da；N_ANALOG 80；SIM_POWER 3 |
| neighbour network（Ours·1，D-L1） | greedy modified cosine，sqrt 强度 | 0.01 Da 取整后 0.0101 Da；去前体峰；top 120；τ 0.6，≥6 匹配峰 |
| extlib rank-1 overlay | greedy sqrt-cosine | 0.015 Da；前体 10 ppm + 0.002 Da；TAU 0.80；≥4 峰；LIB_TAU 0.90 保护 |

注意：GBM ranker（`fit_rankers` / `fit_pv_rankers`）读取的是**预先算好**的训练特征矩阵，analog 相似度是其中的输入特征。

## 2. 评测设计（防泄漏）

- 隐藏测试集为 enveda timsTOF 天然产物（frozen 管线文档；`test.parquet` 只是格式样例）。所以查询一律取 `ingest_lib == enveda-180`（1,153,785 条 timsTOF 谱，182,941 个结构）。
- 结构按 `inchikey14` 划分角色，三者互斥（`extract.py`，seed 20261124）：
  - `cross_query`：2,538 个同时有非 timsTOF 谱的结构（Task A / overlay）；
  - `analog_query`：2,000 个结构；`analog_library`：40,000 个其他结构（Task B/C/edges）。
- Task B/C 中查询结构**不在** analog 库内，模拟新分子；Task C 的候选来自全部训练结构中同分子式者（真值 + 最多 200 个诱饵）。
- 所有差值都是相对管线现设置的**分子级配对 bootstrap**（2,000 次）CI95。

局限：Task B/C 的查询是单条合并 CE 谱（`20,40,60` 优先），管线实际是多谱；analog 库是 40k 子集而非全部 2.5M 谱；Task C 只用 analog 证据，没有管线的其他 30 个特征——**它衡量的是这个特征本身的区分力，不是 LB 分数**。

## 3. 碎片 m/z 误差（`mass_error.py`）

对 400 个 `[M+H]+`/`[M-H]-` 分子各谱 top-20 峰做 sub-formula 注释（C,H,N,O,S,P,Cl 子式，最近理论离子），±30 ppm 内误差直方图在 +0~+4 ppm 处高峰、两侧平坦背景约 128/2 ppm。

| 指标 | 值 |
|---|---|
| 核心中位误差 | +2.51 ppm（MAD 2.60 ppm） |
| 去背景后信号 5%/50%/95% 分位 | −11.0 / +1.6 / +9.5 ppm |
| 绝对误差 q95，m/z 50–150 / 150–300 / 300–600 | 0.0010 / 0.0015 / 0.0028 Da |
| 同分子不同 CE 重复谱最近峰 | 中位 2.8 ppm，q95（|误差|≤10 ppm 的配对）0.0013 Da |

含义：`0.002 Da + 10 ppm`（m/z 100 处 0.003 Da，500 处 0.007 Da）覆盖真实误差且远窄于 0.01 Da。+2.5 ppm 的系统偏移在同仪器库内互相抵消；跨仪器匹配时可考虑校正。

## 4. 实验结果

### 4.1 Task A：跨仪器身份检索（`bench.py --task A`）

14,866 条 timsTOF 查询谱，253,151 条非 timsTOF 库谱，150 万对；2,213 个结构可评估（真值在候选中且候选 ≥2）。按单谱计分。

| 变体 | spectrum MRR | top-1 | Δ vs 管线 [CI95] |
|---|---:|---:|---|
| **pipeline_entropy**（现设置） | 0.9877 | 0.9794 | — |
| cos_sqrt_0.015（overlay 用） | 0.9821 | 0.9699 | −0.0056 [−0.0072, −0.0040] |
| cos_raw_0.01 | 0.9647 | 0.9436 | −0.0230 [−0.0263, −0.0198] |
| entropy_unweighted | 0.9812 | 0.9686 | −0.0065 [−0.0083, −0.0048] |
| ent_tol_10ppm+0.002 | 0.9886 | 0.9813 | +0.0009 [+0.0002, +0.0016] |
| ent_minmz_50 | 0.9892 | 0.9819 | +0.0015 [+0.0007, +0.0023] |
| ent_10ppm_rmprec_modified | 0.9886 | 0.9806 | +0.0009 [−0.0002, +0.0019] |

完整 26 个变体见 `results/taskA.json`。身份检索已饱和；主要结论是 entropy 加权的价值（与 Li et al. 2021 一致）。

### 4.2 Task B：analog 检索（`bench.py --task B`）

2,000 查询 × 54,799 库谱（同极性，|Δ中性质量| ≤ 200 Da），6,643 万对。参考：oracle（窗内最佳 Tanimoto）0.559，随机 0.141。

| 变体 | top-1 Tan | mean top-5 | best top-10 | 加权 FP soft-Tan | Δ top-1 [CI95] |
|---|---:|---:|---:|---:|---|
| **pipeline_entropy** | 0.314 | 0.255 | 0.430 | 0.188 | — |
| cos_sqrt_0.015 | 0.274 | 0.226 | 0.378 | 0.180 | −0.040 [−0.047, −0.033] |
| entropy_unweighted | 0.297 | 0.240 | 0.411 | 0.183 | −0.017 |
| ent_tol_0.02 | 0.302 | 0.243 | 0.417 | 0.180 | −0.012 |
| ent_tol_10ppm+0.002 | 0.323 | 0.261 | 0.438 | 0.193 | +0.009 [+0.006, +0.012] |
| ent_rm_prec | 0.348 | 0.279 | 0.452 | 0.202 | +0.034 [+0.028, +0.040] |
| ent_noshift | 0.305 | 0.250 | 0.411 | 0.197 | −0.009 |
| ent_modified（hybrid） | 0.375 | 0.294 | 0.466 | 0.200 | +0.060 [+0.054, +0.067] |
| ent_modified_rmprec | 0.381 | 0.299 | 0.471 | 0.209 | +0.067 [+0.060, +0.073] |
| **ent_10ppm_rmprec_modified** | **0.382** | **0.302** | **0.474** | **0.215** | **+0.068 [+0.061, +0.075]** |
| cos_sqrt_modified_0.01_rmprec | 0.366 | 0.287 | 0.460 | 0.199 | +0.052 [+0.045, +0.059] |
| ent_nl_0.5（neutral-loss 融合） | 0.319 | 0.264 | 0.453 | 0.197 | +0.005 [−0.001, +0.011] |
| ent_top64 / top128 / centroid / minmz_50 / floor_0 | ≈0.314 | | | | 均不显著 |

机理：`max(unshifted, shifted)` 让每个参考谱只能选"全部碎片按原位"或"全部按中性丢失"之一，而真实类似物通常两类峰兼有（修饰位置之外的碎片不移位，含修饰的碎片移位），一次 one-to-one 分配能同时计入（Watrous et al. 2012；Bittremieux et al. 2022）。前体峰在低 CE 谱中常是最强峰，它只在质量相同时匹配，会压低真类似物、抬高同质量无关结构。

### 4.3 Task C：analog 证据能否给同分异构体正确排序（`task_c.py`）

候选分数 = Σ_top80 sim³ · Tanimoto(候选, analog) / Σ sim³（管线 rank feature `mean_tan` 的形式）。随机排序 MRR 0.181。

| 变体 | MRR | top-1 | Δ MRR [CI95] |
|---|---:|---:|---|
| **pipeline_entropy** | 0.593 | 0.467 | — |
| cos_sqrt_0.01 | 0.522 | 0.379 | −0.071 [−0.086, −0.056] |
| ent_tol_10ppm+0.002 | 0.606 | 0.480 | +0.014 [+0.008, +0.019] |
| ent_rm_prec | 0.663 | 0.538 | +0.070 [+0.057, +0.081] |
| ent_modified | 0.672 | 0.555 | +0.079 [+0.068, +0.090] |
| cos_sqrt_modified_0.01_rmprec | 0.671 | 0.554 | +0.078 [+0.066, +0.091] |
| **ent_10ppm_rmprec_modified** | **0.716** | **0.606** | **+0.124 [+0.111, +0.136]** |

三项改动可叠加（tolerance 单独 +0.014，去前体峰 +0.070，hybrid +0.079，合计 +0.124）。

### 4.4 extlib overlay 校准（`overlay_calibration.py`）

known：库里有真值；novel：移除真值的谱（模拟库外新分子，任何提升都是错的）。≥4 匹配峰。

| 设置 | 正确提升 | 错误提升（known） | 误提升（novel） |
|---|---:|---:|---:|
| **cos_sqrt_0.015，TAU 0.80（现设置）** | 0.868 | 0.012 | 0.046 |
| weighted entropy 0.01 Da，τ 0.65 | **0.937** | 0.006 | 0.040 |
| weighted entropy 0.01 Da，τ 0.70 | 0.909 | **0.005** | **0.029** |
| weighted entropy 0.01 Da，τ 0.80 | 0.658 | 0.003 | 0.008 |

注意 entropy 与 cosine 分数尺度不同，**不能沿用 TAU 0.80**（否则正确提升从 0.868 掉到 0.658）。本库来自 train 内的 GNPS/MassBank/MoNA 等非 timsTOF 谱，与实际 extlib 来源相近但不相同。

### 4.5 neighbour network 边质量（`edges.py`）

600 查询 × 40k 库，1,970 万对，其中仅 281 对 Tanimoto ≥ 0.6。

| 设置 | AUC（Tan≥0.6） | τ0.6 边/查询 | τ0.6 精度 | τ0.7 精度 |
|---|---:|---:|---:|---:|
| network 现设置（modcos，0.01 取整，0.0101 Da） | 0.917 | 41.7 | 0.008 | 0.016 |
| modcos，0.002 Da + 10 ppm | 0.928 | 27.1 | 0.011 | 0.021 |
| cosine（不 shift） | 0.839 | 23.5 | 0.006 | 0.012 |
| modified entropy，10 ppm，去前体峰 | **0.939** | 10.4 | 0.027 | 0.055 |

在 enveda 库中真正的近邻极少，网络边多为弱相关结构。网络层的收益（0.490 → 0.495）可能主要来自 D-L1 的已知结构节点；换 metric 需重调 τ 与 λ。

## 5. 文献与公开实现（DOI 已经 Crossref / arXiv API 核对）

| 文献 | 与本研究的关系 |
|---|---|
| Li Y. et al. Spectral entropy outperforms MS/MS dot product similarity… *Nat Methods* 2021. doi:10.1038/s41592-021-01331-z | 管线使用的 weighted entropy；本研究在 Task A/B 复现其优于 cosine 的结论 |
| Li Y., Fiehn O. Flash entropy search… *Nat Methods* 2023. doi:10.1038/s41592-023-02012-9 | 大库快速 entropy 检索，含 hybrid（fragment + neutral loss）模式；可用于在全 2.5M 谱上做 analog 搜索 |
| Watrous J. et al. Mass spectral molecular networking… *PNAS* 2012. doi:10.1073/pnas.1203689109 | modified cosine 原始定义 |
| Wang M. et al. GNPS… *Nat Biotechnol* 2016. doi:10.1038/nbt.3597 | 分子网络与 extlib 中 GNPS 来源 |
| Bittremieux W. et al. Comparison of cosine, modified cosine, and neutral loss based spectrum alignment… *JASMS* 2022. doi:10.1021/jasms.2c00153 | 发现 modified cosine 对结构相关对最好、neutral-loss 单独无优势；与本研究 Task B 一致 |
| Moorthy A. S. et al. Combining fragment-ion and neutral-loss matching… (hybrid search) *Anal Chem* 2017. doi:10.1021/acs.analchem.7b03320 | hybrid 一次性分配的思路来源 |
| Stein S. E., Scott D. R. Optimization and testing of mass spectral library search algorithms… *JASMS* 1994. doi:10.1016/1044-0305(94)87009-8 | 强度/质量加权 dot product 的经典依据 |
| Huber F. et al. matchms. *JOSS* 2020. doi:10.21105/joss.02411 | CosineGreedy / ModifiedCosine 参考实现（本研究的 greedy 分配与其一致） |
| de Jonge N. F. et al. Reproducible MS/MS library cleaning pipeline in matchms. *J Cheminform* 2024. doi:10.1186/s13321-024-00878-1 | 库清洗（前体峰、噪声）规范 |
| Huber F. et al. Spec2Vec. *PLoS Comput Biol* 2021. doi:10.1371/journal.pcbi.1008724 | 学习型相似度，analog 搜索优于 cosine |
| Huber F. et al. MS2DeepScore. *J Cheminform* 2021. doi:10.1186/s13321-021-00558-4 | 预测结构相似度的学习型谱相似度，下一步可在 Task B 框架评测 |
| de Jonge N. F. et al. MS2Query. *Nat Commun* 2023. doi:10.1038/s41467-023-37446-4 | analog 搜索 + 可靠性重排；思路与 Task C 后置层相同 |
| Bushuiev R. et al. DreaMS. *Nat Biotechnol* 2025. doi:10.1038/s41587-025-02663-3 | 自监督谱表示，可做 analog 嵌入检索 |
| Bushuiev R. et al. MassSpecGym. arXiv:2410.23326 (NeurIPS 2024 D&B) | 检索基准与分子不相交划分方法 |

比赛规则方面：以上改动只用 `train.parquet` 与已挂载的公开数据，推理离线，不引入新外部数据或付费 API。

## 6. 落地方案（按优先级）

1. **analog-evidence 后置层（不需重训，最先做）**：在 FRP5 现有 network 层旁边增加一层，用 `pv_modified_kernel.search_modified_pre`（hybrid entropy，0.002 Da + 10 ppm，去前体峰）在 train 库上为每个测试分子取 top-80 analog，对候选算 Task C 分数，以 `z(−log rank) + λ·z(analog score)` 只重排 2–25 名（保留 rank 1，与网络层相同的安全策略）。λ 先在 enveda-np-examples panel（250 分子）上选，再单次上榜验证。
2. **替换管线 analog kernel 并重算 ranker 训练特征**：收益可能最大，但必须用新 kernel 重新生成 `rank_train` / `pv_rank` 特征矩阵后再拟合 GBM；只在推理端替换会造成特征分布偏移，不建议。
3. **extlib overlay 换成 weighted entropy，τ 0.70**（或 0.65）：改动 1 行函数 + 1 个阈值，三项指标同时改善。
4. 网络层改用 modified entropy / 10 ppm 并重调 τ、λ：证据较弱，排在后面。

不建议继续投入：neutral-loss 谱融合、峰数上限、centroid、min m/z 截断、更宽 tolerance。

## 7. 复现

```bash
.venv/bin/kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data && unzip data/*.zip -d data
python - <<'EOF'   # meta.parquet: train metadata columns only
import pyarrow.parquet as pq
cols=['ingest_lib','inchikey14','instrument_type','adduct','ionization_mode','precursor_mz','precursor_error_ppm','num_peaks','collision_energy_orig']
pq.read_table('data/train.parquet',columns=cols).to_pandas().to_parquet('work/meta.parquet')
EOF
P=research/spectral_matching
.venv/bin/python $P/extract.py --meta work/meta.parquet --out work/sm          # ~6 min, 532,894 spectra
.venv/bin/python $P/mass_error.py work/sm/spectra.parquet work/sm/roles.json $P/results/mass_error.json
.venv/bin/python $P/bench.py --data work/sm --task A --out $P/results/taskA.json   # ~6 min
.venv/bin/python $P/bench.py --data work/sm --task B --n-query 2000 --out $P/results/taskB.json  # ~55 min, 4 CPU
.venv/bin/python $P/task_c.py --data work/sm --out $P/results/taskC.json          # ~15 min
.venv/bin/python $P/edges.py --data work/sm --out $P/results/edges.json
.venv/bin/python $P/overlay_calibration.py --data work/sm --out $P/results/overlay_calibration.json
.venv/bin/python -m pytest -q tests/test_spectral_matching.py
```

峰值内存约 9 GB（Task B 1,500+ 查询时），4 核 CPU 总耗时约 1.5 小时。

## 8. 上线测试：FRP5 + analog 后置层（2026-11-24）

- 实现：`analog_layer.py` + `notebook_cell_analog.py`，作为 FRP5（0.495）之后的唯一新增 cell（Ours · 4），其余 cell 字节不变；只重排 2–25 名，rank 1 与候选集合不变，任何异常/超时保留原文件。λ = 0.35。
- 离线端到端代理（公开 250 分子 NP panel，`panel_eval.py`，`results/panel_analog_layer.json`）：λ ≤ 0.35 时三种情形差值都在 ±0.0003 以内（CI 含 0）；λ ≥ 0.75 在"结构不在池中"情形变差（−0.003）。该 panel 已饱和：真值在 2–25 名、可被本层移动的分子每种情形只有 14–19 个，且该情形的真值全来自未能映射结构的来源，因此**离线没有证明提升，也没有发现损害**。
- Kaggle：notebook `xiaoyuzhoux120/casmi26-nightly-analog-20261124` v1，commit 运行通过（本层 334 s，400 分子）；提交 58847301，public **0.495**，与 FRP5 原版 0.495 相同（2026-11-25 查分）。结论：该后置层在 public 榜上**没有可测收益**（也没有损害）；不建议作为默认，可在 λ 更大或结合重算 ranker 特征时再测。public 榜约 125 个分子，±0.004 约等于一个分子从第 2 名到第 1 名，差 0.001–0.003 应视为噪声。
