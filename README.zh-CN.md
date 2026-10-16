# CALIPER：局部性感知的本地—云端协作 RAG

本仓库整理 CALIPER 论文对应的公开代码、轻量文档、图表和复现入口，目标是让读者能看懂代码结构、实验协议和复现边界。

当前稿件：`2026-10-16-submission-v1`，对应 Overleaf `f72ec11`，正文 10 页、31 条参考文献。Figure 2 采用 `paper/workflow.pdf`。

## 方法概览

CALIPER 将云端问题指导与本地证据驱动回答分开：

1. 云端模型接收问题并返回指导；
2. 知识库、检索和证据驱动回答留在本地；
3. source router 根据问题来源选择专用检索配置；
4. 可选执行分类器将预测为本地可回答的问题直接交给本地模型。

这里的“本地”边界不等于形式化隐私保证。问题和云端返回的指导仍可能包含敏感信息，因此论文和代码统一使用“locality-aware local-cloud RAG”，不宣称差分隐私、密码学保护或法规合规。

## 当前公开范围

公开：代码、配置模板、轻量结果摘要、图表、论文源文件和复现说明。

不公开：原始数据、检索证据、模型权重、微调 checkpoint、FAISS 缓存、云端响应、逐题预测、日志、API Key 和大型归档。

当前目录是当前冻结协议的公开版本。原始数据、权重、缓存、云端响应、逐题预测、日志和归档均不公开；逐表命令模板见 `docs/REPRODUCTION_COMMANDS.md`。历史脚本统一放在 `legacy/`，不作为默认入口；当前核心入口的默认模型名称已与冻结协议对齐，表格聚合脚本仍需要私有路径和结果文件。

## 目录

```text
根目录核心脚本       CALIPER、本地/云端基线、指标计算和检索缓存构建
query_clf/            执行分类器训练辅助脚本
model_router/         source router 训练辅助脚本
scripts/              当前表格、消融、统计和审计入口
work/                 预留的当前运行溯源目录
legacy/               历史版本和修复脚本；带日期的运行溯源在 `legacy/work/`
external/Adaptive-RAG 外部对比方法代码及其许可证
retrieve_fine_tuning/公开的训练脚本模板，不含训练数据和权重
figures/              Figure 1–3 导出文件及源文件
tables/               Table 1–7 源片段及编号说明
paper/                论文源文件、PDF、Highlights 和补充说明
configs/              冻结协议 JSON 与环境变量模板
data/、models/        占位目录；数据和模型不随仓库发布
results/、logs/       占位目录；不提交原始输出
docs/                 协议、边界和发布说明
```

## 当前冻结协议

- 本地模型：`Qwen3.5-9B`；
- 云端参考模型：`Qwen3.7-plus`；
- 检索初始化：`bge-base-en-v1.5` + `bge-reranker-large`；
- 数据集：HotpotQA、MuSiQue、2WikiMultiHopQA；
- 执行分类器阈值：HotpotQA `0.974`、MuSiQue `0.978`、2Wiki `0.977`；
- Table 5 是按每题分别测量检索和生成后组合得到的 estimated end-to-end latency，不是一次完整在线链路的整体实测；
- 统计使用逐题配对 bootstrap、10,000 次重采样和 percentile 95% 区间。

具体数字以 `paper/Supplementary_Configuration_Notes.md`、论文源文件和冻结归档为准，不能用不同模型、prompt 或缓存直接重跑后替代。

逐表运行命令、私有输入契约和“实时运行/结果聚合”的边界见 `docs/REPRODUCTION_COMMANDS.md` 与 `configs/current_protocol.json`。

## 复现入口

| 论文项目 | 主要入口 |
|---|---|
| Table 2/4 | `scripts/eval_table2_final.py`、`scripts/eval_ablation_final.py`、`hybrid_predict_separate.py` |
| Table 3 | `scripts/eval_table3_full.py` |
| Table 5 | `scripts/measure_table4_final.py`、`scripts/bootstrap_table5_current_20260930.py` |
| Table 6 | `scripts/eval_table6_*.py` 和 `classifier_analysis/` |
| Table 7 | 检索缓存构建脚本和 `scripts/eval_table6_*.py` |
| Figure 3 | 分类器阈值分析脚本 |

运行这些入口需要私有数据、模型、检索缓存和本地/云端服务。仓库不自动下载，也不会把这些资源上传到 GitHub。

## 安装与密钥

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env            # Windows: copy .env.example .env
```

真实 API Key 只能放在本地 `.env`，不能提交。推送前必须做 secret scan 和大文件扫描。

代码仓库地址：<https://github.com/voidance13/CALIPER>
