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

"""Build evaluation JSONL from messages JSONL.

Replicates the prompt-building logic of ACTDataset.build_prompt_dataset
(base_dataset.py) as a standalone script so we can generate eval files
for any dataset without instantiating the full training pipeline.

Input:  messages JSONL (output of convert_pacific.py or convert_abg_coqa.py)
Output: eval JSONL with fields required by ucp_clarify.scripts.evaluate:
        input_text, output_text, dialogue_policy, gold_target, gold_trajectory,
        chosen_policy, rejected_policy, chosen, rejected
"""

import argparse
import json

from ucp_clarify.data.constants import (
    _MODEL_ROLE,
    _PREFERENCE_SYSTEM_INSTRUCTION,
    _SYSTEM_ROLE,
    _USER_ROLE,
)


def process_context(content: str) -> str:
    return "[context]\n{}\n[conversation]\n".format(content)


def get_next_turn(i: int, messages: list[dict], clarify: bool):
    if i + 1 >= len(messages):
        return "", None

    if not clarify:
        output_text = "Assistant: {}\n".format(messages[i + 1]["content"])
        answer = None
    else:
        output_text = "Assistant: {}\n".format(messages[i + 1]["content"])
        if i + 2 < len(messages):
            user_reply = messages[i + 2]
            output_text += "User: {}\n".format(
                user_reply["content"].replace("\n", " ")
            )
            if i + 3 < len(messages):
                final_turn = messages[i + 3]
                final_line = "Assistant: {}\n".format(
                    final_turn["content"].replace("\n", " ")
                )
                output_text += final_line
                answer = final_line
            else:
                answer = "This question was not answered by the Assistant."
        else:
            answer = "This question was not answered by the Assistant."

    return output_text, answer


def build_eval_instances(samples: list[dict]) -> list[dict]:
    system_instruction = _PREFERENCE_SYSTEM_INSTRUCTION
    instances = []

    for sample in samples:
        input_text = "{}\n".format(system_instruction)

        for i, message in enumerate(sample["messages"]):
            if message["role"] == _SYSTEM_ROLE:
                input_text += process_context(message["content"])

            elif message["role"] == _USER_ROLE:
                input_text += "User: {}\n".format(
                    message["content"].replace("\n", " ")
                )

                if message.get("requires_clarification", False):
                    output_text, answer = get_next_turn(
                        i, sample["messages"], clarify=True
                    )
                    dialogue_policy = "CLARIFY"
                    gold_trajectory = input_text + output_text
                    gold_target = answer
                    chosen_policy = "CLARIFY"
                    rejected_policy = "ANSWER"
                    chosen_response = output_text
                    rejected_response = gold_target
                else:
                    output_text, answer = get_next_turn(
                        i, sample["messages"], clarify=False
                    )
                    dialogue_policy = "ANSWER"
                    gold_trajectory = input_text + output_text
                    gold_target = output_text
                    chosen_policy = "ANSWER"
                    rejected_policy = "CLARIFY"
                    chosen_response = output_text
                    rejected_response = "PLACEHOLDER"

                instance = {
                    "input_text": input_text,
                    "output_text": output_text,
                    "dialogue_policy": dialogue_policy,
                    "gold_target": gold_target,
                    "gold_trajectory": gold_trajectory,
                    "chosen_policy": chosen_policy,
                    "rejected_policy": rejected_policy,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                }
                instances.append(instance)
                input_text = input_text + output_text

    return instances


def main():
    parser = argparse.ArgumentParser(
        description="Build eval JSONL from messages JSONL."
    )
    parser.add_argument(
        "--messages_path", type=str, required=True,
        help="Input messages JSONL (from convert_*.py).",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output eval JSONL path.",
    )
    args = parser.parse_args()

    with open(args.messages_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    instances = build_eval_instances(samples)

    with open(args.output_path, "w", encoding="utf-8") as out:
        for inst in instances:
            json.dump(inst, out, ensure_ascii=False)
            out.write("\n")

    policy_counts = {}
    for inst in instances:
        p = inst["dialogue_policy"]
        policy_counts[p] = policy_counts.get(p, 0) + 1

    print(f"Built {len(instances)} eval instances -> {args.output_path}")
    print(f"  Policy distribution: {policy_counts}")


if __name__ == "__main__":
    main()
