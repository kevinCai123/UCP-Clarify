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

"""Convert Abg-CoQA dataset into ACT messages JSONL format.

Mirrors convert_pacific.py but handles the Abg-CoQA schema:
  { id, story, target_turn, history_turns, ambiguity, clarification_turn, source }

Output: one JSON-lines file where each line is
  { "messages": [ {role, content, requires_clarification?}, ... ] }
matching the format that ACTDataset.build_prompt_dataset expects.
"""

import argparse
import json

from ucp_clarify.data.constants import _MODEL_ROLE, _SYSTEM_ROLE, _USER_ROLE


def wrap_answer(text: str) -> str:
    """Wrap a plain-text answer in the ['...'] list format used by PACIFIC."""
    return "['" + text.replace("'", "\\'") + "']"


def select_gold_branch(target_answer: str, branches: list[dict]) -> dict:
    """Pick the clarification branch whose org_ans matches the target answer."""
    target_norm = target_answer.strip().lower()
    for branch in branches:
        if branch["org_ans"].strip().lower() == target_norm:
            return branch
    return branches[0]


def convert_item(item: dict) -> dict | None:
    """Convert one Abg-CoQA item into a messages dict."""
    story = item["story"].replace("\n", " ").strip()
    messages = []

    messages.append({
        "role": _SYSTEM_ROLE,
        "content": "[context]\n{}\n".format(story),
    })

    for turn in item["history_turns"]:
        messages.append({
            "role": _USER_ROLE,
            "content": turn["question"],
            "requires_clarification": False,
        })
        messages.append({
            "role": _MODEL_ROLE,
            "content": wrap_answer(turn["answer"]),
        })

    target = item["target_turn"]

    if item["ambiguity"] == "non_ambiguous":
        messages.append({
            "role": _USER_ROLE,
            "content": target["question"],
            "requires_clarification": False,
        })
        messages.append({
            "role": _MODEL_ROLE,
            "content": wrap_answer(target["answer"]),
        })
    else:
        ct = item.get("clarification_turn", {})
        if not ct or "question" not in ct or "answers" not in ct:
            return None

        branch = select_gold_branch(target["answer"], ct["answers"])

        messages.append({
            "role": _USER_ROLE,
            "content": target["question"],
            "requires_clarification": True,
        })
        messages.append({
            "role": _MODEL_ROLE,
            "content": wrap_answer(ct["question"]),
        })
        messages.append({
            "role": _USER_ROLE,
            "content": branch["clr_ans"],
            "requires_clarification": False,
        })
        messages.append({
            "role": _MODEL_ROLE,
            "content": wrap_answer(branch["org_ans"]),
        })

    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Convert Abg-CoQA to ACT messages format.")
    parser.add_argument("--path", type=str, required=True, help="Path to raw Abg-CoQA JSON.")
    parser.add_argument("--results_path", type=str, required=True, help="Output JSONL path.")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw["data"]
    skipped = 0

    with open(args.results_path, "w", encoding="utf-8") as out:
        for item in items:
            converted = convert_item(item)
            if converted is None:
                skipped += 1
                continue
            json.dump(converted, out, ensure_ascii=False)
            out.write("\n")

    total = len(items)
    print(f"Converted {total - skipped}/{total} items ({skipped} skipped) -> {args.results_path}")


if __name__ == "__main__":
    main()
