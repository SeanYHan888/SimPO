# Setup With uv

## 1) Prerequisites

- Linux cluster with CUDA drivers installed.
- `uv` installed (`uv --version`).
- Access to your training model and dataset repos on Hugging Face (if private/gated).

## 2) Create/Sync Environment

From the repo root:

```bash
cd /Users/seanmacbook/Research/dpo/SimPO
uv sync
```

This installs the core dependencies from `pyproject.toml` and the pinned versions in `uv.lock`.

## 3) Install Optional Training Extras

For a typical GPU training stack:

```bash
uv sync --extra all
```

Or install only what you need:

```bash
uv sync --extra wandb
uv sync --extra quant
uv sync --extra deepspeed
uv sync --extra flash-attn
```

### Flash-Attn If Build Is Slow

Fastest safe path (skip flash-attn entirely):

```bash
uv sync --extra wandb --extra quant --extra deepspeed
```

Then run training with:

```bash
--attn_implementation=sdpa --torch_dtype=bfloat16
```

If you still want flash-attn, install it only after your final torch version is already installed:

```bash
uv pip install --only-binary=:all: flash-attn
# If no matching wheel exists for your torch/cuda/python combo, it fails fast.

uv pip install --force-reinstall --no-build-isolation flash-attn
```

This avoids most ABI breakages caused by building flash-attn against a different torch/cuda runtime.

## 4) Run SimPO Training

```bash
uv run accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-v2.yaml
```

Swap the config file in `training_configs/` for your target model setup.

## 5) Tiny Smoke Run (Validated)

Use a short single-process run with LoRA to validate dependencies and script wiring before full training:

```bash
uv run accelerate launch --num_processes 1 \
  scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-v2.yaml \
  --max_steps=5 \
  --do_eval=false \
  --per_device_train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --logging_steps=1 \
  --save_steps=5 \
  --attn_implementation=sdpa \
  --use_peft=true \
  --lora_r=8 \
  --lora_alpha=16 \
  --lora_target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max_length=512 \
  --max_prompt_length=256 \
  --output_dir=outputs/smoke-llama-3-8b-instruct-simpo-v2 \
  --run_name=smoke-llama-3-8b-instruct-simpo-v2
```

## 6) Resume Check (Recommended)

Use this to verify checkpoint resume works before expensive multi-GPU runs:

```bash
uv run accelerate launch --num_processes 1 \
  scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-v2.yaml \
  --max_steps=22 \
  --resume_from_checkpoint=outputs/smoke-llama-3-8b-instruct-simpo-v2/checkpoint-5 \
  --attn_implementation=sdpa \
  --use_peft=true \
  --lora_r=8 \
  --lora_alpha=16 \
  --lora_target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max_length=512 \
  --max_prompt_length=256 \
  --output_dir=outputs/smoke-llama-3-8b-instruct-simpo-v2 \
  --run_name=smoke-llama-3-8b-instruct-simpo-v2-resume
```

Important: keep LoRA/model-shape args identical between initial run and resume.

## 7) Diagnose Flash-Attn ABI Mismatch

If you see an error like `undefined symbol ... c10_cuda_check_implementation ...`, run:

```bash
uv run python scripts/diagnose_flash_attn.py
```

That script prints torch/cuda/flash-attn import status and explicitly flags ABI mismatch.

## 8) Re-lock Dependencies (when changing versions)

```bash
uv lock
```

Then commit both `pyproject.toml` and `uv.lock`.
