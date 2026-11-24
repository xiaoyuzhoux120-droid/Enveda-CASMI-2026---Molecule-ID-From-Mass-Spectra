# CASMI 2026 研究环境配置记录

记录日期：2026-11-24（Claude Projects 云端线程实测）。本文件只记录实际运行过的结果；未能验证的项目明确标为“未验证/待授权”。

## 1. 快速开始

```bash
bash scripts/setup_env.sh                  # 创建 .venv（Python 3.12）并按 lock 文件安装，CPU 版 PyTorch
.venv/bin/python scripts/verify_env.py     # 全部环境检查（只读，不提交、不上传）
```

GPU 机器：`TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_env.sh`（按本机 CUDA 版本选择官方 wheel 源；此时不用 lock 文件里的 `+cpu` 版本）。

## 2. 云端运行环境（实测）

| 项目 | 结果 |
|---|---|
| 系统 | Ubuntu 24.04.5，Linux x86_64 |
| CPU / 内存 | 4 核 / 约 16 GB，无 swap |
| GPU | 无（没有 `nvidia-smi`，`torch.cuda.is_available()` 为 False） |
| 磁盘 | 根分区可写约 61 GB（每会话固定额度） |
| Python | 系统 3.13.18；项目使用 `/usr/bin/python3.12`（3.12.3）建 `.venv`，与 Kaggle wheel 一致 |
| 已有工具 | `uv`、`git`、`gh`（已有 GitHub 令牌），未重复安装 |
| 会话性质 | 容器是临时的，闲置后回收；`.venv`、数据和模型不会保留。Default 云端环境已保存一个 setup 脚本：新容器启动时自动在本仓库建 `.venv`（分支合并前按 `requirements.txt` + 研究包安装，合并后调用 `scripts/setup_env.sh`）。该脚本已在本会话的仓库副本上实测，但要等真正新建的线程才能证实 |

### 网络（云端环境 “Default”，受限模式 + 允许包管理器）

本次为 Default 环境新增的允许域名（经你在审批卡片上批准）：
`www.kaggle.com`、`api.kaggle.com`、`export.arxiv.org`、`api.semanticscholar.org`、`api.crossref.org`、`download.pytorch.org`、`download-r2.pytorch.org`。

实测：PyPI、PyTorch CPU 源、Kaggle 网站与 MCP、arXiv、Crossref 可访问；Semantic Scholar 可连通但匿名请求持续返回 429（共享限流）。`huggingface.co`、`zenodo.org` 未放行（目前没有确定用途，需要时再加）。

注意：你另有一个名为 `enveda-casmi-2026` 的云端环境（已允许 Kaggle 主机），但本项目线程实际运行在 Default 环境，所以配置写在 Default 上，对你所有使用 Default 的云端会话生效。

## 3. 已安装的 Python 包（`.venv`，精确版本见 `requirements-lock.txt`）

| 包 | 版本 | 说明 |
|---|---|---|
| numpy / pandas / pyarrow / scipy | 2.5.5 / 3.1.1 / 26.0.0 / 1.18.1 | 满足原 `requirements.txt` |
| scikit-learn / joblib | 1.9.1 / 1.6.0 | 与 `requirements-ml.txt` 固定版本一致 |
| torch | 2.15.0+cpu | 官方 CPU wheel 源 |
| rdkit | 2026.03.3 | **与 Kaggle 评分器固定版本完全一致** |
| matchms | 1.0.0 | |
| pyteomics | 5.0.2 | |
| optuna | 5.1.0 | |
| jupyterlab / ipykernel | 4.6.6 | |
| pytest | 9.2.0 | |
| kaggle (CLI) / kagglehub | 2.2.5 / 1.0.3 | 官方 |

依赖文件：
- `requirements.txt`、`requirements-ml.txt`：原文件，**未修改**。
- `requirements-research.txt`：新增，研究环境的直接依赖（RDKit 固定 2026.3.3）。
- `requirements-lock.txt`：新增，本次实际解析出的全部 147 个包的精确版本。

### 评估后暂不安装的包

用 `uv pip install --dry-run` 在当前环境检查了兼容性：

| 包 | 结论 | 证据 / 理由 |
|---|---|---|
| MassSpecGym | 不装进主环境 | 会把 rdkit 降到 2023.9.4、numpy 到 1.24.4、torch 到 2.3.0、matchms 到 0.26.2，破坏与评分器一致的 RDKit。若需其基准数据或基线，另建独立 venv（并需放行 huggingface.co）。 |
| MS2DeepScore | 暂不装 | 会把 pandas 3.1.1 降到 2.3.3。适合做 class-1 谱库检索的学习型相似度，需要时在独立 venv 评估，其预训练权重需确认训练数据来源合规（见第 5 节规则）。 |
| Spec2Vec | 暂不装 | 无冲突（gensim 4.4.0），但对当前 MRR 提升路线用途不明确。 |
| PyTorch Geometric | 暂不装 | 无冲突；现有 `casmi_ml/direct_models.py` 已自带图批处理，暂无必须用途。 |
| HF Transformers | 暂不装 | 无冲突；只有做 de novo SMILES 生成（class 3）时才有明确用途。 |

## 4. 工具与外部访问

### Kaggle 官方 MCP Server

- 官方文档 https://www.kaggle.com/docs/mcp ：远程地址 `https://www.kaggle.com/mcp`（HTTP 传输），支持 OAuth 2.0 或 `Authorization: Bearer <KGAT 开头的 API token>`。
- 实测（`scripts/check_kaggle_mcp.py`，只做握手和列工具）：初始化成功，服务端 `Kaggle.Web 1.0.0.0`，**76 个工具**。
- 匿名可用（已实测）：比赛元数据 `get_competition`、比赛页面 `list_competition_pages`（规则/评估/数据说明）、讨论区 `list_competition_topics` / `list_topic_messages`。
- 需要登录（已实测返回 `Unauthenticated`）：`list_competition_data_files`、`download_competition_data_file`、`search_notebooks`。
- 工具中包含写操作：`submit_to_competition`、`create_code_competition_submission`、`save_notebook`、`create_notebook_session`、`upload_dataset_file`、`create_model`、`update_*` 等。**未经你许可不会调用这些工具**；`scripts/` 下的脚本都不调用任何写工具。
- 项目级配置 `.mcp.json`（已提交，不含密钥）：

  ```json
  {"mcpServers": {"kaggle": {"type": "http", "url": "https://www.kaggle.com/mcp",
    "headers": {"Authorization": "Bearer ${KAGGLE_API_TOKEN:-}"}}}}
  ```

  令牌只从环境变量 `KAGGLE_API_TOKEN` 读取。实测 `claude mcp list` 显示该服务器为 “Pending approval”：Claude Code 要求用户本人批准项目级 MCP 服务器，我没有代为批准。
- 已知问题：Claude Code 的 OAuth 流程与 Kaggle 授权服务器存在不兼容报告（anthropics/claude-code issue #84839，token 交换时报 `client_secret_basic authentication requires a client_secret`），因此推荐用 API token 头方式。
- 实测小坑：Kaggle 对 User-Agent 为 `curl/8.5.0` 这类完整 curl 版本串的请求返回 404，而 Python requests、Claude Code 的 UA 都正常。用 curl 手工调试时请加 `-A curl/8`。

### Kaggle CLI / KaggleHub（MCP 的可靠替代）

`kaggle` 2.2.5 与 `kagglehub` 1.0.3 已安装。CLI 支持 `KAGGLE_API_TOKEN` 环境变量、`~/.kaggle/access_token` 文件或 `kaggle auth login`（OAuth）。当前云端**没有配置任何 Kaggle 凭证**，所以认证类命令未能验证。

### GitHub

- 仓库 `xiaoyuzhoux120-droid/Enveda-CASMI-2026---Molecule-ID-From-Mass-Spectra`（公开 fork）已以推送权限挂到本线程，并**保存到项目**：之后新建的线程会自动带上它（可在项目设置里移除）。
- 本次改动全部在分支 `env-setup-20261124` 上，通过 PR 提交，未改动 `main`。

### 学术检索

- 使用官方免费 API，未安装第三方 MCP：`scripts/lit_search.py`（arXiv API、Semantic Scholar Graph API、Crossref REST API）。
- 实测：arXiv、Crossref 返回结果；Semantic Scholar 匿名请求 429（全局共享限流）。可申请免费 API key 后设为 `SEMANTIC_SCHOLAR_API_KEY` 环境变量提升限额（可选）。
- Claude 自带的网页搜索与网页读取在云端线程中可用（本次已用于查官方文档）。

## 5. 比赛信息（经 Kaggle 官方 MCP 读取，2026-11-24）

- 页面：https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra （Featured，$50,000，5,887 队）
- 时间：报名/接受规则与组队截止 **2026-12-07 23:59 UTC**；最终提交截止 **2026-12-14 23:59 UTC**。
- 评估：MRR@25。双方 SMILES 都经 RDKit 2026.03.3 互变异构体规范化后取 InChIKey 前 14 位比较；只算第一个正确候选；元数据字段名显示为 “CASMI: Mean Average Precision”，但评估页定义是 MRR@25。
- 提交：Notebook-only 代码赛；CPU 或 GPU Notebook ≤ 9 小时；**必须关闭互联网**；输出 `submission.csv`（`molecule_id,smiles`，每个分子最多 25 个用 `;` 连接的 SMILES）；每天最多 5 次，最终选 2 个。
- 数据：`train.parquet`（约 250 万张谱、约 27.5 万结构，18 列）、`test.parquet`（可见部分是训练集样例，重跑时替换为约 400 分子/1,500 张 timsTOF 谱的隐藏集，12 列）、`sample_submission.csv`。数据许可 CC BY-NC 4.0，仅限比赛与非商业学术用途，**不得向未接受规则的人分发**。
- 外部数据：允许免费公开的外部数据和预训练模型。主办方 2026-10-17 讨论帖（#750330）澄清：允许带非商业条款（如 CC-BY-NC）的开源数据；**NIST 明确禁止**（含其衍生模型权重）；非开源、非免费或专有数据违规。讨论中有人追问预训练权重是否需要公开训练代码和数据，截至读取时未见主办方最终答复，使用他人权重前请自行确认。
- 获奖方须以 MIT 等 OSI 许可开源代码。

## 6. 环境验证结果（`scripts/verify_env.py` 实际输出，完整 JSON 见 `results/env_check_20261124.json`）

| # | 检查 | 结果 | 实测细节 |
|---|---|---|---|
| 1 | Kaggle MCP 可用 | 通过（匿名） | 76 个工具；读取到比赛截止时间与指标 |
| 1b | Kaggle CLI 认证 | **未通过：未配置凭证** | 等你按第 8 节配置 |
| 2 | 比赛数据合法获取 | **未验证** | 需要凭证 + 你本人在网页上接受规则；数据文件尚未下载 |
| 3 | RDKit 解析与指纹 | 通过 | 版本 2026.03.3；两种葡萄糖写法规范化后 InChIKey14 均为 `WQZGKKKJIJFFOK`（复现评估页示例）；Morgan 2048 位指纹、Tanimoto 正常 |
| 4 | MatchMS 处理与相似度 | 通过 | default_filters + 归一化；CosineGreedy 0.9973（4 峰匹配），+14 Da 平移谱 ModifiedCosine 0.9973 |
| 5 | PyTorch 训练 | 通过 | CPU 4 线程，合成数据 MLP 30 轮：loss 0.6921 → 0.3227，位准确率 0.862 |
| 6 | 项目现有代码 | 通过 | `pytest tests`：29 passed，1 skipped（unittest 同样 30 个用例 OK） |
| 7 | 新线程可复现 | 部分验证（云端 setup 脚本已保存，待新线程证实） | 在全新目录从仓库文件运行 `setup_env.sh` 后 `verify_env.py` 得到相同 7/9 结果；尚未在另一个真正新建的云端线程中验证 |

说明：第一次 PyTorch 检查失败过一次，原因是我写的测试断言过严（取的是第 1 轮之后的 loss 作基准），不是环境问题；改为与训练前 loss 比较后通过。

GPU 训练脚本 `scripts/train_fp_gpu.py` 只用**合成的 parquet（8 个真实 SMILES + 随机峰）**在 CPU 上做了冒烟测试，确认数据读取、指纹标签、分子不相交划分、训练与保存流程能跑通；dev Tanimoto 约 0.01，这是合成数据，不代表任何真实性能。

## 7. CPU 验证与 GPU 训练

- 云端没有 GPU：在云端只做小规模 CPU 验证（`--max-molecules 500 --epochs 2` 级别）。
- 本地 GPU：`.venv/bin/python scripts/train_fp_gpu.py --data data/train.parquet --max-molecules 60000 --out artifacts/fp_gpu_run1`（CUDA 时自动用 BF16 autocast）。
- 原有 GPU 实验代码 `casmi_ml/scale_experiment.py`（RTX 5070 上跑过）保持不变，输出目录固定，重新实验请按 `docs/EXPERIMENTS.md` 新建目录。
- Kaggle GPU：`kaggle_gpu_training/` 是私有训练脚本模板（开 GPU、开互联网用于装 RDKit、只写 `/kaggle/working`），**没有推送**，由你审核后自己 push（步骤见该目录 README）。它与关网的提交 Notebook 分开。

## 8. 需要你手动完成的事项

1. **接受比赛规则**（12 月 7 日前）：浏览器登录 Kaggle → 打开比赛页面 → Rules → “I Understand and Accept”。不接受无法下载数据。
2. **生成 Kaggle API token**：Kaggle → 右上角头像 → Settings → API → “Generate New Token”，复制（以 `KGAT` 开头）。不要发到聊天里。
3. **把 token 交给云端线程**（二选一，都不进 Git）：
   - 推荐：claude.ai → 本项目 → Project settings → Cloud environment → Default 旁的齿轮 → Environment variables，添加 `KAGGLE_API_TOKEN=<你的 token>`。只对之后启动或恢复的会话生效。
   - 注意：该处环境变量以明文保存在环境配置中，对使用 Default 环境的所有会话可见。若在意，可只在本地电脑使用 token（`~/.kaggle/access_token`，权限 600），云端只做不需数据的工作。
4. **本地电脑**：`export KAGGLE_API_TOKEN=...` 写入 shell 配置，或 `kaggle auth login`。Claude Code 本地可用 `claude mcp add --transport http kaggle https://www.kaggle.com/mcp --header "Authorization: Bearer $KAGGLE_API_TOKEN"`（user 级，不进仓库）。
5. **批准项目 MCP 服务器**：在带本仓库的 Claude Code 中运行 `claude`，按提示批准 `.mcp.json` 里的 `kaggle`。这一步我无法代你完成。
6. **合并 PR**：审阅分支 `env-setup-20261124` 的 PR 后由你决定是否合并。
7. 可选：申请免费 Semantic Scholar API key，设为 `SEMANTIC_SCHOLAR_API_KEY`；需要 MassSpecGym/Hugging Face 数据时再放行 `huggingface.co`。

完成第 1–3 步后，在新线程运行 `bash scripts/setup_env.sh && .venv/bin/python scripts/verify_env.py`，`kaggle_cli_auth` 应通过；然后再下载数据，`competition_data_present` 通过即全部验证完成。

## 9. 安全与已知限制

- 不在仓库、日志或普通配置中写任何密钥；`.gitignore` 已忽略 `kaggle.json`、`.env*`、数据与模型文件。脚本只打印“凭证是否存在”。
- 没有提交比赛结果、没有发布或推送 Notebook、没有修改团队信息、没有调用付费服务。
- 云端容器临时；比赛数据约 3 GB，下载后只在当前会话有效，且按规则不得外传。
- 云端 4 核 / 16 GB 内存，载入完整 train.parquet 的全部列可能接近内存上限，建议按列读取或分批。
- Kaggle MCP 部分工具需要登录才能用；OAuth 在 Claude Code 中有已知问题。
