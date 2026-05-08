# coding=utf-8
# Copyright 2025 Songlin Cai.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a comparison markdown table from evaluation metrics across methods.

Usage:
    python -m ucp_clarify.scripts.compare_results --conv 25
    python -m ucp_clarify.scripts.compare_results --conv 50
    python -m ucp_clarify.scripts.compare_results --conv 100
    python -m ucp_clarify.scripts.compare_results --conv 25 --compare_with 10 50
"""

import argparse
import json
import datetime
import pathlib


METHODS = {
    "SFT": "sft_metrics.json",
    "ACT Original": "act_original_metrics.json",
    "ACT Dynamic Beta": "act_dynamic_beta_metrics.json",
    "ACT Beta Filter": "act_beta_filter_metrics.json",
    "ACT Beta DPO Full": "act_beta_dpo_full_metrics.json",
}

METRICS_ORDER = [
    "Accuracy",
    "Weighted F1",
    "Macro F1",
    "Binary F1",
    "Binary Recall",
    "Binary Precision",
    "Turn-Level DROP F1",
    "Trajectory-level DROP F1",
    "Post-Clarification DROP F1",
    "Post-Clarification Count",
]


def load_metrics(eval_dir: pathlib.Path) -> dict:
    data = {}
    for name, fname in METHODS.items():
        fpath = eval_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data[name] = json.load(f)
        else:
            print(f"WARNING: {fpath} not found")
            data[name] = {}
    return data


def format_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    elif isinstance(v, int):
        return str(v)
    return str(v)


def build_main_table(all_data: dict) -> list[str]:
    header = "| Metric | " + " | ".join(METHODS.keys()) + " |"
    sep = "|---|" + "|".join(["---:"] * len(METHODS)) + "|"
    lines = [header, sep]

    for metric in METRICS_ORDER:
        vals = [all_data[name].get(metric, "N/A") for name in METHODS]
        numeric = [v for v in vals if isinstance(v, (int, float))]
        best = max(numeric) if numeric else None
        cells = []
        for v in vals:
            s = format_value(v)
            if best is not None and v == best:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    return lines


def build_delta_table(all_data: dict, other_data: dict, conv: int, other_conv: int) -> list[str]:
    header = "| Metric | " + " | ".join(METHODS.keys()) + " |"
    sep = "|---|" + "|".join(["---:"] * len(METHODS)) + "|"
    lines = [
        f"\n## Compared With {other_conv}-Conversation Results\n",
        f"Values below are `{conv}-conv - {other_conv}-conv` deltas. Positive means {conv}-conv scored higher.\n",
        header,
        sep,
    ]

    for metric in METRICS_ORDER:
        cells = []
        for name in METHODS:
            v_cur = all_data[name].get(metric, None)
            v_other = other_data.get(name, {}).get(metric, None)
            if isinstance(v_cur, (int, float)) and isinstance(v_other, (int, float)):
                delta = v_cur - v_other
                s = f"{delta:+.4f}" if isinstance(delta, float) else f"{delta:+d}"
            else:
                s = "N/A"
            cells.append(s)
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate comparison table from eval metrics")
    parser.add_argument("--conv", type=int, required=True, help="Number of conversations (e.g. 25, 50, 100)")
    parser.add_argument("--compare_with", type=int, nargs="*", default=None,
                        help="Other conversation counts to compare deltas against (default: all other available)")
    parser.add_argument("--eval_base", type=str, default="runs/table1_seq/eval",
                        help="Base directory for eval results")
    args = parser.parse_args()

    eval_dir = pathlib.Path(args.eval_base) / str(args.conv)
    if not eval_dir.exists():
        print(f"ERROR: Eval directory {eval_dir} does not exist")
        return

    all_data = load_metrics(eval_dir)

    lines = [
        f"# {args.conv}-Conversation Comparison\n",
        f"Generated: {datetime.datetime.now().astimezone().isoformat()}\n",
        "Bold denotes the highest numeric value in each metric row.\n",
    ]
    lines.extend(build_main_table(all_data))

    compare_targets = args.compare_with
    if compare_targets is None:
        base = pathlib.Path(args.eval_base)
        compare_targets = sorted(
            int(d.name) for d in base.iterdir()
            if d.is_dir() and d.name.isdigit() and int(d.name) != args.conv
        )

    for other_conv in compare_targets:
        other_dir = pathlib.Path(args.eval_base) / str(other_conv)
        other_data = load_metrics(other_dir)
        if any(other_data.values()):
            lines.extend(build_delta_table(all_data, other_data, args.conv, other_conv))

    output = "\n".join(lines) + "\n"
    out_path = eval_dir / f"comparison_{args.conv}conv.md"
    with open(out_path, "w") as f:
        f.write(output)
    print(f"Comparison table written to {out_path}")
    print()
    print(output)


if __name__ == "__main__":
    main()
