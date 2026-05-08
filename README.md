# UCP-Clarify

Reference implementation for UCP-Clarify. The Python package is named
`ucp_clarify`, so commands are run with `python -m ucp_clarify...`.

This is research code and is not an officially supported product.

## Acknowledgement

This project builds upon **Learning to Clarify (ACT)** by The Google Research
Authors. The original codebase is available at:
<https://github.com/google-research/google-research/tree/master/learning_to_clarify>

Our modifications introduce dynamic-beta DPO scheduling, soft preference
filtering, and evaluation across different dpo variances on top of the original
ACT framework.

## Citation

If you use this code, please cite both works:

```bibtex
@inproceedings{
chen2025learning,
title={Learning to Clarify: Multi-turn Conversations with Action-Based Contrastive Self-Training},
author={Maximillian Chen and Ruoxi Sun and Tomas Pfister and Sercan O Arik},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=SIE6VFps9x}
}
```

The UCP-Clarify paper is currently **under review**. Citation details will be
added upon publication.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
Files originating from the original ACT codebase retain the copyright notice
of The Google Research Authors. See [NOTICE](NOTICE) for details.

## Setup Environment

Assume this repository is under `/home/myuser/staging/UCP-Clarify`.

```bash
cd /home/myuser/staging/UCP-Clarify
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
export PYTHONPATH=/home/myuser/staging/UCP-Clarify/src
```

Set credentials for the generative model provider. The examples below use
Gemini, matching the original ACT example style:

```bash
export GOOGLE_API_KEY=YourApiKey
export HF_TOKEN=YourHFTokenId
export MODEL_API=gemini-2.0-flash-001
```

You can use OpenAI instead:

```bash
export OPENAI_API_KEY=YourApiKey
export OPENAI_MODEL=gpt-4o-mini
export MODEL_API=openai
```

## Download PACIFIC Dataset

```bash
cd /home/myuser/staging
git clone https://github.com/dengyang17/PACIFIC
```

## Convert PACIFIC Dataset to UCP-Clarify Format

```bash
cd /home/myuser/staging/UCP-Clarify

python3 -m ucp_clarify.scripts.convert_pacific \
  --path=/home/myuser/staging/PACIFIC/data/pacific/train.json \
  --results_path=/home/myuser/staging/train.jsonl

python3 -m ucp_clarify.scripts.convert_pacific \
  --path=/home/myuser/staging/PACIFIC/data/pacific/validation.json \
  --results_path=/home/myuser/staging/validation.jsonl
```

## Sample Data for an Example Run

For quick smoke tests, sample a small train and validation set. For actual
training, use at least 50 training conversations and the full validation set.

```bash
cd /home/myuser/staging
shuf -n 50 train.jsonl > train_50samples.jsonl
shuf -n 50 validation.jsonl > validation_50samples.jsonl
```

## Generate Preference Data

This step converts conversation JSONL into preference-pair JSONL. It writes
`train_preference.jsonl` and `validation_preference.jsonl`.

```bash
mkdir -p /home/myuser/staging/output_dir/preference_data

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.generate_preference \
  --output_dir=/home/myuser/staging/output_dir/preference_data \
  --train_path=/home/myuser/staging/train_50samples.jsonl \
  --validation_path=/home/myuser/staging/validation_50samples.jsonl \
  --preference_model_id=$MODEL_API \
  --icl_examples=10
```

## Run SFT on Qwen 2.5 3B Instruct Model from Hugging Face

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/sft

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.run_sft \
  --output_dir=/home/myuser/staging/output_dir/model_output/sft \
  --train_path=/home/myuser/staging/output_dir/preference_data/train_preference.jsonl \
  --validation_path=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_token=$HF_TOKEN \
  --policy_model_id=Qwen/Qwen2.5-3B-Instruct \
  --preference_model_id=$MODEL_API \
  --is_preference_task=True \
  --num_train_epochs=4 \
  --bf16=False
```

For multi-GPU training, use Accelerate:

```bash
accelerate launch \
  --config_file src/ucp_clarify/utils/deepspeed_zero3.yaml \
  -m ucp_clarify.scripts.run_sft \
  --output_dir=/home/myuser/staging/output_dir/model_output/sft \
  --train_path=/home/myuser/staging/output_dir/preference_data/train_preference.jsonl \
  --validation_path=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_token=$HF_TOKEN \
  --policy_model_id=Qwen/Qwen2.5-3B-Instruct \
  --preference_model_id=$MODEL_API \
  --is_preference_task=True \
  --num_train_epochs=4
```

## Evaluate SFT

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/sft_eval

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/sft \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/sft_eval/sft_eval.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/sft_eval/sft_eval_samples.json
```

## Run Vanilla DPO on Qwen 2.5 3B Instruct SFT Model

The original ACT training mode is exposed as `act_original`. In this release we
refer to that baseline as vanilla DPO.

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/vanilla_dpo

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.run_act \
  --output_dir=/home/myuser/staging/output_dir/model_output/vanilla_dpo \
  --train_path=/home/myuser/staging/output_dir/preference_data/train_preference.jsonl \
  --validation_path=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/sft \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --preference_model_id=$MODEL_API \
  --is_preference_task=True \
  --experiment_mode=act_original \
  --beta=0.05 \
  --num_train_epochs=4
```

## Evaluate Vanilla DPO

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/vanilla_dpo_eval

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/vanilla_dpo \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/vanilla_dpo_eval/vanilla_dpo_eval.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/vanilla_dpo_eval/vanilla_dpo_eval_samples.json
```

## Run UCP-Clarify with Dynamic Beta Only

This mode enables the dynamic-beta mechanism and leaves filtering disabled.

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/ucp_clarify_dynamic_beta

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.run_act \
  --output_dir=/home/myuser/staging/output_dir/model_output/ucp_clarify_dynamic_beta \
  --train_path=/home/myuser/staging/output_dir/preference_data/train_preference.jsonl \
  --validation_path=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/sft \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --preference_model_id=$MODEL_API \
  --is_preference_task=True \
  --experiment_mode=act_dynamic_beta \
  --beta=0.05 \
  --beta_dpo_alpha=0.6 \
  --beta_dpo_ema_gamma=0.9 \
  --beta_dpo_min_beta=0.001 \
  --beta_dpo_max_beta=0.15 \
  --beta_dpo_warmup_steps=0 \
  --num_train_epochs=4
```

## Evaluate UCP-Clarify with Dynamic Beta Only

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/ucp_clarify_dynamic_beta_eval

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_dynamic_beta \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_dynamic_beta_eval/ucp_clarify_dynamic_beta_eval.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_dynamic_beta_eval/ucp_clarify_dynamic_beta_eval_samples.json
```

## Run UCP-Clarify with Filter Only

This mode keeps a fixed DPO beta and enables soft filtering. The filter
downweights high-discrepancy outliers; `CLARIFY` samples are protected.

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/ucp_clarify_filter

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.run_act \
  --output_dir=/home/myuser/staging/output_dir/model_output/ucp_clarify_filter \
  --train_path=/home/myuser/staging/output_dir/preference_data/train_preference.jsonl \
  --validation_path=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/sft \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --preference_model_id=$MODEL_API \
  --is_preference_task=True \
  --experiment_mode=act_beta_filter \
  --beta=0.05 \
  --beta_dpo_filter_ratio=0.2 \
  --beta_dpo_ema_gamma=0.9 \
  --num_train_epochs=4
```

## Evaluate UCP-Clarify with Filter Only

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/ucp_clarify_filter_eval

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_filter \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_filter_eval/ucp_clarify_filter_eval.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_filter_eval/ucp_clarify_filter_eval_samples.json
```

## Run Complete UCP-Clarify

This mode enables both dynamic beta and soft filtering.

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/ucp_clarify_full

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.run_act \
  --output_dir=/home/myuser/staging/output_dir/model_output/ucp_clarify_full \
  --train_path=/home/myuser/staging/output_dir/preference_data/train_preference.jsonl \
  --validation_path=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/sft \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --preference_model_id=$MODEL_API \
  --is_preference_task=True \
  --experiment_mode=act_beta_dpo_full \
  --beta=0.05 \
  --beta_dpo_alpha=0.6 \
  --beta_dpo_filter_ratio=0.2 \
  --beta_dpo_ema_gamma=0.9 \
  --beta_dpo_min_beta=0.001 \
  --beta_dpo_max_beta=0.15 \
  --beta_dpo_warmup_steps=0 \
  --num_train_epochs=4
```

## Evaluate Complete UCP-Clarify

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/ucp_clarify_full_eval

cd /home/myuser/staging/UCP-Clarify
python3 -m ucp_clarify.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_full \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_full_eval/ucp_clarify_full_eval.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_full_eval/ucp_clarify_full_eval_samples.json
```

## Optional Evaluation Limit

Use `--max_eval_samples` for a quick partial evaluation.

```bash
python3 -m ucp_clarify.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_full \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_full_eval/ucp_clarify_full_eval_100.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/ucp_clarify_full_eval/ucp_clarify_full_eval_100_samples.json \
  --max_eval_samples=100
```
