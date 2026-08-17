# PEFT-QLoRA example for fine-tuning

PEFT-QLoRA fine-tuning example using the
[FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
dataset and the lightweight
[deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
model, designed to run on a free T4 GPU (16 GB VRAM).

## How It Works

**Dataset formatting**

The `medical-o1-reasoning-SFT` dataset has three fields: `Question`, `Complex_CoT`, and `Response`.
The notebook formats these into the model's native chat template using its real special tokens
(`<\uff5cUser\uff5c>`, `<\uff5cAssistant\uff5c>`, `<\uff5cend\u2581of\u2581sentence\uff5c>` — the U+FF5C
characters), plus the ` thinking`/` response` reasoning tags.
Forcing the model to start with ` thinking` at generation time keeps it in "thinking" mode, matching how
the base model was trained.

**Efficient training**

QLoRA quantizes the base model to 4-bit, cutting memory drastically. A 1.5B-class model drops from
~3.6 GB (FP16) to ~0.9 GB (4-bit), so training fits comfortably on a T4.

> **GPU precision note**: the Tesla T4 is Turing (compute capability 7.5) and has no native bfloat16
> tensor cores. `bf16` is emulated and 2-4x slower (or errors on some torch versions). The notebook
> therefore uses `fp16` (`bnb_4bit_compute_dtype=torch.float16`, `fp16=True`). On Ampere+ GPUs switch
> to `bf16` if you prefer.

**Stack**

- `trl` — `SFTTrainer`/`SFTConfig` for the supervised fine-tuning loop
- `peft` — `LoraConfig` + `prepare_model_for_kbit_training`
- `bitsandbytes` — 4-bit NF4 quantization (with double quantization)
- `transformers`, `datasets`, `accelerate` — model loading, data loading, distributed training

## Usage

Open `qlora_v1.ipynb` in Colab (Runtime → Change runtime type → T4 GPU) and run top to bottom.
The last cell runs a generation test on a held-out question.

## Changes in v2

See the changelog table in the notebook's final cell. Highlights:

- Fixed chat format: used `<|user|>`/`<|end|>` (ASCII) tokens that are **not** in this model's
  vocabulary; replaced with the model's real full-width special tokens.
- Fixed T4 precision: `bf16` → `fp16`.
- Removed unnecessary `trust_remote_code=True`.
- Updated to the current `trl` API (`SFTConfig`, `dataset_text_field=None`, `max_length`).
- Fixed adapter save path (`trainer.model.save_pretrained` instead of the unwrapped base model).
- Added `double_quant`, `save_total_limit`, seed, and an inference smoke-test cell.