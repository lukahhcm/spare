#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from mnms.constants import TOOL_METADATA


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def replace_node_with_placeholder(text):
    pattern = r"<node-\d+>\.\w+"
    matches = re.findall(pattern, text)
    format_args = []
    for match in matches:
        node_id = int(match[6 : match.index(">")])
        arg_name = match.split(".")[-1]
        format_args.append((node_id, arg_name))
    return re.sub(pattern, "{}", text), format_args


def nodes_to_code(nodes):
    code = []
    for node in nodes:
        node_name = f"output{node['id']}"
        fn_name = node["name"].replace(" ", "_")
        arg_strs = []
        for arg_name, arg_value in node["args"].items():
            if isinstance(arg_value, str):
                if arg_value.startswith("<node-") and " " not in arg_value:
                    node_id = int(arg_value[6 : arg_value.index(">")])
                    node_key = arg_value[arg_value.index(".") + 1 :]
                    arg_value = f"output{node_id}['{node_key}']"
                elif "<node-" in arg_value:
                    placeholder_str, node_infos = replace_node_with_placeholder(arg_value)
                    node_values = [f"output{node_id}['{node_key}']" for node_id, node_key in node_infos]
                    arg_value = placeholder_str.format(*node_values).replace('"', '\\"')
                    arg_value = f'f"{arg_value}"'
                else:
                    arg_value = f'"{arg_value.replace(chr(34), chr(92) + chr(34))}"'
            else:
                arg_value = f'"{arg_value}"'
            arg_strs.append(f"{arg_name}={arg_value}")
        code.append(f"{node_name} = {fn_name}({', '.join(arg_strs)})")
    return "\n".join(code)


def extract_func_info_from_single_line(code):
    try:
        tree = ast.parse(code)
        result = {"output_var": "", "func_name": "", "args": {}}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                result["output_var"] = node.targets[0].id
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                    result["func_name"] = node.value.func.id
                for keyword in node.value.keywords:
                    result["args"][keyword.arg] = ast.unparse(keyword.value)
                break
        if not result["output_var"] or not result["func_name"]:
            return None
    except Exception:
        return None
    return result


def extract_func_info_from_code(code):
    info = []
    for line in code.split("\n"):
        result = extract_func_info_from_single_line(line.strip())
        if result:
            info.append(result)
    return info


def get_code_labels(code, include_arg_names=False, include_arg_values=False, edge_as_label=False, sep="--"):
    parsed = extract_func_info_from_code(code)
    labels = []
    for i, func_info in enumerate(parsed):
        if "func_name" not in func_info:
            continue
        if edge_as_label:
            if i + 1 >= len(parsed) or "func_name" not in parsed[i + 1]:
                continue
            label = [parsed[i]["func_name"], parsed[i + 1]["func_name"]]
        else:
            label = [func_info["func_name"]]
            if include_arg_names:
                for arg_name, arg_value in sorted(func_info["args"].items()):
                    if include_arg_values:
                        if isinstance(arg_value, str):
                            arg_value = arg_value.lower()
                        label.append(f"{arg_name}={arg_value}")
                    else:
                        label.append(arg_name)
        labels.append(sep.join(label))
    return labels


def flatten_labels(gt_labels, pred_labels, categories=None):
    union = list(set(gt_labels + pred_labels))
    if categories:
        cat_index = {cat: i + 1 for i, cat in enumerate(categories)}
        gt_flat = [cat_index.get(label, 0) if label in gt_labels else 0 for label in union]
        pred_flat = [cat_index.get(label, 0) if label in pred_labels else 0 for label in union]
    else:
        gt_flat = [1 if label in gt_labels else 0 for label in union]
        pred_flat = [1 if label in pred_labels else 0 for label in union]
    return gt_flat, pred_flat


def prf_from_counts(tp, fp, fn):
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return round(100 * precision, 2), round(100 * recall, 2), round(100 * f1, 2)


def compute_prf(pairs, categories=None, average="binary"):
    if average == "macro" and categories:
        per_label = {i + 1: [0, 0, 0] for i in range(len(categories))}
        for gt_labels, pred_labels in pairs:
            gt_flat, pred_flat = flatten_labels(gt_labels, pred_labels, categories)
            for g, p in zip(gt_flat, pred_flat):
                if g == 0 and p == 0:
                    continue
                labels = set([x for x in (g, p) if x != 0])
                for label in labels:
                    if g == label and p == label:
                        per_label[label][0] += 1
                    elif g == label and p != label:
                        per_label[label][2] += 1
                    elif g != label and p == label:
                        per_label[label][1] += 1
        vals = [prf_from_counts(*counts) for counts in per_label.values()]
        return {
            "precision": round(sum(v[0] for v in vals) / len(vals), 2),
            "recall": round(sum(v[1] for v in vals) / len(vals), 2),
            "f1": round(sum(v[2] for v in vals) / len(vals), 2),
        }

    tp = fp = fn = 0
    for gt_labels, pred_labels in pairs:
        gt_flat, pred_flat = flatten_labels(gt_labels, pred_labels, categories)
        for g, p in zip(gt_flat, pred_flat):
            if average == "binary" or not categories:
                if g == 1 and p == 1:
                    tp += 1
                elif g == 0 and p == 1:
                    fp += 1
                elif g == 1 and p == 0:
                    fn += 1
            else:
                if g != 0 and p == g:
                    tp += 1
                elif p != 0 and p != g:
                    fp += 1
                elif g != 0 and p != g:
                    fn += 1
    precision, recall, f1 = prf_from_counts(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1}


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = old
    return dp[-1]


def sequence_ratio(a, b):
    if not a and not b:
        return 1.0
    dist = edit_distance(a, b)
    return 1.0 - dist / max(len(a), len(b), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-file", required=True)
    ap.add_argument("--preds-file", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--per-sample-jsonl", default=None)
    args = ap.parse_args()

    gt_data = list(read_jsonl(args.gt_file))
    pred_data = list(read_jsonl(args.preds_file))
    id2pred = {sample["id"]: sample for sample in pred_data}
    categories = [tool.replace(" ", "_") for tool in TOOL_METADATA.keys()]
    labelers = {
        "tool_macro": partial(get_code_labels),
        "argname": partial(get_code_labels, include_arg_names=True),
        "argvalue": partial(get_code_labels, include_arg_names=True, include_arg_values=True),
        "edge": partial(get_code_labels, edge_as_label=True),
        "plan_tool": partial(get_code_labels),
        "plan_argname": partial(get_code_labels, include_arg_names=True),
        "plan_argvalue": partial(get_code_labels, include_arg_names=True, include_arg_values=True),
        "tool": partial(get_code_labels),
    }

    label_pairs = defaultdict(list)
    invalid = []
    missing = []
    per_sample = []
    status_counts = Counter()
    for gt in gt_data:
        pred = id2pred.get(gt["id"])
        if pred is None:
            missing.append(gt["id"])
            pred_code = ""
            pred_nodes = []
            pred_status = "missing"
        else:
            pred_status = pred.get("status", "")
            status_counts[pred_status] += 1
            try:
                pred_nodes = pred["prediction"]
                pred_code = nodes_to_code(pred_nodes)
            except Exception as exc:
                invalid.append({"id": gt["id"], "error": repr(exc)})
                pred_nodes = []
                pred_code = ""
        gt_nodes = ast.literal_eval(gt["plan_str"])
        gt_code = nodes_to_code(gt_nodes)
        sample_record = {"id": gt["id"], "status": pred_status, "gt_steps": len(gt_nodes), "pred_steps": len(pred_nodes)}
        for name, labeler in labelers.items():
            gt_labels = labeler(gt_code)
            pred_labels = labeler(pred_code)
            label_pairs[name].append((gt_labels, pred_labels))
            if name in {"plan_tool", "plan_argname", "plan_argvalue"}:
                sample_record[f"{name}_exact"] = pred_labels == gt_labels
            if name == "tool":
                sample_record["tool_seq_similarity"] = round(100 * sequence_ratio(pred_labels, gt_labels), 2)
                sample_record["gt_tools"] = gt_labels
                sample_record["pred_tools"] = pred_labels
        per_sample.append(sample_record)

    metrics = {}
    metrics.update({f"tool_macro_{k}": v for k, v in compute_prf(label_pairs["tool_macro"], categories, "macro").items()})
    metrics.update({f"argname_{k}": v for k, v in compute_prf(label_pairs["argname"], None, "binary").items()})
    metrics.update({f"argvalue_{k}": v for k, v in compute_prf(label_pairs["argvalue"], None, "binary").items()})
    metrics.update({f"edge_{k}": v for k, v in compute_prf(label_pairs["edge"], None, "binary").items()})
    for name in ["plan_tool", "plan_argname", "plan_argvalue"]:
        acc = sum(pred == gt for gt, pred in label_pairs[name]) / len(label_pairs[name])
        metrics[f"{name}_accuracy"] = round(100 * acc, 2)
    tool_ned = 100 * (1 - sum(sequence_ratio(pred, gt) for gt, pred in label_pairs["tool"]) / len(label_pairs["tool"]))
    metrics["tool_normalized_edit_distance"] = round(tool_ned, 2)
    metrics.update(
        {
            "num_gt": len(gt_data),
            "num_pred": len(pred_data),
            "num_missing": len(missing),
            "num_invalid_prediction": len(invalid),
            "pred_status_counts": dict(status_counts),
        }
    )

    records = [{"metric": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v} for k, v in metrics.items()]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(records)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.per_sample_jsonl:
        with open(args.per_sample_jsonl, "w", encoding="utf-8") as f:
            for row in per_sample:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in records:
        print(f"{row['metric']}: {row['value']}")


if __name__ == "__main__":
    main()
