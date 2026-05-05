# UC-ACT

Reference implementation for UC-ACT. The Python package is named `uc_act`, so
commands are run with `python -m uc_act...`.

This is research code and is not an officially supported product.

## Setup Environment

Assume this repository is under `/home/myuser/staging/UC-ACT`.

```bash
cd /home/myuser/staging/UC-ACT
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
export PYTHONPATH=/home/myuser/staging/UC-ACT/src
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

## Convert PACIFIC Dataset to UC-ACT Format

```bash
cd /home/myuser/staging/UC-ACT

python3 -m uc_act.scripts.convert_pacific \
  --path=/home/myuser/staging/PACIFIC/data/pacific/train.json \
  --results_path=/home/myuser/staging/train.jsonl

python3 -m uc_act.scripts.convert_pacific \
  --path=/home/myuser/staging/PACIFIC/data/pacific/validation.json \
  --results_path=/home/myuser/staging/validation.jsonl
```

## Sample Data for an Example Run

For quick smoke tests, sample a small train and validation set. For actual
training, use at least 50 training conversations and the full validation set.

```bash
cd /home/myuser/staging
shuf -n 20 train.jsonl > train_20samples.jsonl
shuf -n 20 validation.jsonl > validation_20samples.jsonl
```

## Generate Preference Data

This step converts conversation JSONL into preference-pair JSONL. It writes
`train_preference.jsonl` and `validation_preference.jsonl`.

```bash
mkdir -p /home/myuser/staging/output_dir/preference_data

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.generate_preference \
  --output_dir=/home/myuser/staging/output_dir/preference_data \
  --train_path=/home/myuser/staging/train_20samples.jsonl \
  --validation_path=/home/myuser/staging/validation_20samples.jsonl \
  --preference_model_id=$MODEL_API \
  --icl_examples=10
```

## Run SFT on Qwen 2.5 3B Instruct Model from Hugging Face

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/sft

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.run_sft \
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
  --config_file src/uc_act/utils/deepspeed_zero3.yaml \
  -m uc_act.scripts.run_sft \
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

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.evaluate \
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

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.run_act \
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

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/vanilla_dpo \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/vanilla_dpo_eval/vanilla_dpo_eval.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/vanilla_dpo_eval/vanilla_dpo_eval_samples.json
```

## Run UC-ACT with Dynamic Beta Only

This mode enables the dynamic-beta mechanism and leaves filtering disabled.

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/uc_act_dynamic_beta

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.run_act \
  --output_dir=/home/myuser/staging/output_dir/model_output/uc_act_dynamic_beta \
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

## Evaluate UC-ACT with Dynamic Beta Only

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/uc_act_dynamic_beta_eval

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/uc_act_dynamic_beta \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/uc_act_dynamic_beta_eval/uc_act_dynamic_beta_eval.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/uc_act_dynamic_beta_eval/uc_act_dynamic_beta_eval_samples.json
```

## Run UC-ACT with Filter Only

This mode keeps a fixed DPO beta and enables soft filtering. The filter
downweights high-discrepancy outliers; `CLARIFY` samples are protected.

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/uc_act_filter

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.run_act \
  --output_dir=/home/myuser/staging/output_dir/model_output/uc_act_filter \
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

## Evaluate UC-ACT with Filter Only

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/uc_act_filter_eval

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/uc_act_filter \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/uc_act_filter_eval/uc_act_filter_eval.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/uc_act_filter_eval/uc_act_filter_eval_samples.json
```

## Run Complete UC-ACT

This mode enables both dynamic beta and soft filtering.

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/uc_act_full

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.run_act \
  --output_dir=/home/myuser/staging/output_dir/model_output/uc_act_full \
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

## Evaluate Complete UC-ACT

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/uc_act_full_eval

cd /home/myuser/staging/UC-ACT
python3 -m uc_act.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/uc_act_full \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/uc_act_full_eval/uc_act_full_eval.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/uc_act_full_eval/uc_act_full_eval_samples.json
```

## Optional Evaluation Limit

Use `--max_eval_samples` for a quick partial evaluation.

```bash
python3 -m uc_act.scripts.evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/uc_act_full \
  --action_model_id=$MODEL_API \
  --simulator_model_id=$MODEL_API \
  --intent_model_id=$MODEL_API \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/uc_act_full_eval/uc_act_full_eval_100.json \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/uc_act_full_eval/uc_act_full_eval_100_samples.json \
  --max_eval_samples=100
```
