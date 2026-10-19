# CASMI26 0.456~0.465 突破复现：guhongbin/casmi26-v465-v15-pop015

- **源工程**: [`guhongbin/casmi26-v465-v15-pop015`](https://www.kaggle.com/code/guhongbin/casmi26-v465-v15-pop015)
- **作者**: `Hongbin Gu (guhongbin)`
- **Kaggle scriptVersionId**: `359783615`
- **公榜成绩**: MRR@25 **0.456**（标题代号 `v465 v15 pop015`，位列前沿梯队）
- **演进路径**: 
  `0.440 (Canqiang E006)` -> `0.441 (v7c)` -> `0.446 (v11)` -> `0.451 (v12/v13)` -> `0.455 (v14)` -> **`0.456 (v15 pop015)`**

---

## 一、 核心技术创新点与提分机制深度拆解

相比此前的 0.440 基线，该版本通过五项关键技术增量，成功将公榜成绩推高至 **0.456+**：

### 1. N1A GLACIER 合成代理转化（N1A Proxies for Blind Molecules）
* **痛点**: 在原有流水线中，GLACIER 仅支持 `[M+H]+` 单质子加合物，导致约 20%~25% 的非 `[M+H]+` 分子处于 GLACIER 盲区（只能回退至相对弱势的 ICEBERG）。
* **创新机制**: 引入 Seyit Kaan Gunes 的 N1A 方案：
  - 对没有 `[M+H]+` 谱图但具备 `[M+NH4]+`（铵加合物）或 `[M-H2O+H]+`（脱水加合物）的分子，构建**合成 `[M+H]+` 代理谱（Proxies）**；
  - 严格剔除母离子前体损失碎片区域（$m/z \ge \text{precursor} - 1.5$ Da），并将前体校正为理论 $[M+H]^+$ 质荷比；
  - 将生成的代理谱图送入 GLACIER 深度模拟器，并通过多碰撞能量合并（Merged-CE）进行前向打分；
* **效果**: 彻底攻克 GLACIER 盲区，显著提升了非质子化分子的 Top 25 准确率。

### 2. 热度先验超参微调（Popularity Prior: `POP_LAM = 0.15`）
* **机制**: Ahmed / E006 中原设定 `POP_LAM = 0.25`；
* **调优**: 作者通过网格搜索发现，过高的 Popularity 权重会导致模型偏向于文献常见分子，对冷门特异性天然产物产生抑制。将 `POP_LAM` 从 0.25 下调至 **0.15**，使得质谱特征打分与热度先验达到更优的平衡点。

### 3. Engine 2 候选库大幅扩增（Pool Expansion to 1.14M Structures）
* 新增接入 `franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints`，额外注入 **356,820** 个高质量 PubChemLite 结构；
* 将原有 ChEBI/LIPID MAPS 替换为 `sergiomalv/casmi-chebi-lmsd-structures`，新增 **98,433** 个结构；
* 使主候选池扩充至 **1,137,233** 个结构，显著拉高了候选池覆盖上限（Recall Ceiling）。

### 4. 零证据分子的 MetFrag-lite 碎片物理重排（Frag Rescore）
* 对既没有 GLACIER 也没有有限 ICEBERG 打分的分子，自动启用未经训练的物理碎裂打分器 `frag_rescore.py`；
* 通过重原子图 1~2 键断裂的理论中性质量匹配，对融合列表计算质谱解释强度（$\lambda = 0.5$），进行物理证据兜底。

### 5. Metric-Identity E2R2 对冲插入（Engine-2 Rank-2 Hedge）
* 采用官方 `metric_identity` 规范（剥离立体化学后的 InChIKey-14 + 重原子组成）；
* 将 Engine 2 排序最高的候选强制插入到最终提交列表的 **第 2 位（Rank 2）**；
* 为 Engine 1 的潜在判错提供了坚实的排名对冲保护。

---

## 二、 挂载数据集清单（Kaggle 运行必备）

要在 Kaggle 上完整复现并执行此 Notebook，需要挂载以下 16 个输入数据源：

### 1. 竞赛数据
* `enveda-CASMI26-molecule-id-mass-spectra` (Competition Data)

### 2. 核心特征与模型权重（15 个 Datasets）
1. `ahmedberatozer/casmi26-candidate-ranker-training-features`
2. `prvsiyan/casmi26-fp-models-v2` (`CASMI26 fingerprint models (single + merged)`)
3. `dmitriigluzdov/casmi26-pubchem-popularity-prior`
4. **`franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints`**（新候选指纹库）
5. `ahmedberatozer/casmi26-simulated-ranker-rows`
6. `ahmedberatozer/casmi26-fpnet-full1`
7. `ahmedberatozer/casmi26-glacier`
8. `ahmedberatozer/casmi26-iceberg`
9. `ahmedberatozer/casmi26-pubchem-tier`
10. `ahmedberatozer/casmi26-v2-pool`
11. `ahmedberatozer/casmi26-v3-models`
12. `ahmedberatozer/casmi26-v4b-models`
13. **`sergiomalv/casmi-chebi-lmsd-structures`**（新 ChEBI/LMSD 结构）
14. `aidensong123/casmi26-coconut-202609` (`COCONUT 2.0 candidates + fingerprints (CASMI26)`)
15. `metric/rdkit-2026-3-3-wheel`

---

## 三、 本地交付物清单

* `casmi26-v465-v15-pop015.ipynb`: 完整提取的 Jupyter Notebook，格式严谨，开箱即用。
* `notebook.py`: 逐 Cell 导出、经过 AST 语法检验的 Python 脚本。
* `SHA256SUMS`: 完整哈希指纹校验清单。
