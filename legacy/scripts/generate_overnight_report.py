import json
from pathlib import Path


def pct(value):
    return f"{100 * value:.2f}"


classifier = json.load(open("classifier_analysis/classifier_error_analysis.json"))
reflection = json.load(open("reflection_analysis/reflection_analysis.json"))
caliper_sig = json.load(open("significance_analysis/caliper_paired_bootstrap.json"))
always_metrics = json.load(open("always_multi_runs/always_multi_full_metrics.json"))
always_sig = json.load(open("always_multi_runs/always_multi_paired_bootstrap.json"))
integrity = json.load(open("always_multi_runs/always_multi_integrity.json"))

names = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "2wikimultihopqa": "2WikiMultiHopQA"}
lines = [
    "# CALIPER 夜间补充实验报告（2026-08-23）", "",
    "## 1. Always-Multi（Adaptive C 分支全量）", "",
    "> 定义：所有题强制走当前 Adaptive-RAG 适配实现的 C 分支；base Top-5 + 两次聚焦子查询 + 同题候选语料 BM25 补证据 + 最终短答案。最大检索次数 3。不是官方 IRCoT 原仓逐参数复现。", "",
    "| 数据集 | n | SBERT | Hit | P | R | F1 | 95% CI | 平均延迟(s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for dataset in names:
    row = always_metrics[dataset]
    lines.append(f"| {names[dataset]} | {row['n']} | {row['SBERT']:.2f} | {row['Hit']:.2f} | {row['P']:.2f} | {row['R']:.2f} | {row['F1']:.2f} | [{row['F1_CI95'][0]:.2f}, {row['F1_CI95'][1]:.2f}] | {row['latency_mean_seconds']:.3f} |")

lines += ["", "### 配对显著性", "", "| 数据集 | 对照 | ΔF1 | 95% CI | p（双侧） |", "|---|---|---:|---:|---:|"]
for dataset in names:
    for baseline in ("adaptive", "naive"):
        row = always_sig[dataset][f"always_multi_vs_{baseline}"]
        lines.append(f"| {names[dataset]} | {baseline} | {row['delta_F1']:+.2f} | [{row['delta_CI95'][0]:+.2f}, {row['delta_CI95'][1]:+.2f}] | {row['p_two_sided']:.6f} |")

lines += ["", "## 2. 分类器误差分析", "", "| 数据集 | 模型工件 | Accuracy | easy recall | hard recall | Macro-F1 | 预测 easy 占比 |", "|---|---|---:|---:|---:|---:|---:|"]
for dataset in names:
    for key, label in (("deployed_table2_router", "现存路由实现"), ("newer_9b_labeled_router", "9B 标注版")):
        row = classifier[dataset][key]
        macro = (row["per_class"]["easy"]["f1"] + row["per_class"]["hard"]["f1"]) / 2
        lines.append(f"| {names[dataset]} | {label} | {pct(row['accuracy'])} | {pct(row['per_class']['easy']['recall'])} | {pct(row['per_class']['hard']['recall'])} | {pct(macro)} | {100 * row['predicted_distribution']['easy'] / row['n']:.2f} |")
lines += [
    "", "结论：现存路由实现偏向 easy，hard recall 仅 8.46% / 25.98% / 11.14%。9B 标注版改善 HotpotQA、MuSiQue 的类别平衡，但 2Wiki accuracy 降至 69.35%，不能统一替换。高置信错误集中在关系链、比较问句和时间/地点组合问句；9B 概率接近 1 的错例较多，单靠置信阈值不能完成校准。", "",
    "历史 Table 5 的 71.86 / 60.20 / 86.88 无法由当前保存模型与当前标注审计集直接复算；本次审计值不覆盖论文表，需先恢复当时验证集和模型哈希。", "",
    "## 3. Reflection 配对分析", "", "| 数据集 | 冻结 V0 方案 | 配对 n | 改写率 | ΔF1 | 95% CI | p（双侧） | 改善/伤害/持平 |", "|---|---|---:|---:|---:|---:|---:|---:|"
]
selected = {"hotpotqa": "reflA_three_vote", "musique": "v0_2stream", "2wikimultihopqa": "v0_opt_run01"}
for dataset, variant in selected.items():
    row = reflection[dataset][variant]
    ci = row["paired_delta_CI95"]
    lines.append(f"| {names[dataset]} | {variant} | {row['paired_n']} | {pct(row['answer_changed_rate'])}% | {100 * row['paired_delta_f1']:+.2f} | [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] | {row['p_two_sided']:.6f} | {row['changed_improved']}/{row['changed_hurt']}/{row['changed_tied']} |")
lines += [
    "", "Reflection 最终工件只保存 question/pred，未保存 evaluator trace、reflection_iterations 或逐次延迟；因此不能从现有数据可靠恢复反思轮数。", "",
    "## 4. CALIPER core 对基线配对显著性", "", "| 数据集 | 对照 | 配对 n | CALIPER F1（配对子集） | 对照 F1 | ΔF1 | 95% CI | p（双侧） |", "|---|---|---:|---:|---:|---:|---:|---:|"
]
for dataset in names:
    for baseline in ("adaptive", "naive", "always_multi"):
        row = caliper_sig[dataset][f"caliper_vs_{baseline}"]
        lines.append(f"| {names[dataset]} | {baseline} | {row['n']} | {row['left_F1']:.2f} | {row['right_F1']:.2f} | {row['delta_F1']:+.2f} | [{row['delta_CI95'][0]:+.2f}, {row['delta_CI95'][1]:+.2f}] | {row['p_two_sided']:.6f} |")
lines += [
    "", "配对检验使用逐题 SQuAD F1、10,000 次 paired bootstrap、seed=42。HotpotQA CALIPER 工件逐题均值 71.30，与表中 73.45 的历史汇总口径不一致；显著性只引用可复核逐题值。MuSiQue 排除 14 条缺失或重复问题。", "",
    "## 5. 完整性与产物", "",
]
for dataset in names:
    row = integrity[dataset]
    lines.append(f"- {names[dataset]}：n={row['n']}，全 C 路由={row['all_route_C']}，空答案={row['empty']}，错误={row['errors']}，SHA256=`{row['sha256']}`")
lines += ["", "远程冻结目录：`/mnt/data/caliper/code/CALIPER/always_multi_runs/`、`classifier_analysis/`、`reflection_analysis/`、`significance_analysis/`。"]

Path("always_multi_runs/overnight_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
