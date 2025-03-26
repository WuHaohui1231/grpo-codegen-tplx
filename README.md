# grpo-codegen-tplx
## Quick Start
1. Create environment
2. Run `pip install -r requirements.txt`
3. Run GRPO training with `deepspeed train.py --deepspeed zero3_offload.json`

**Note:**
1. It's normal that the training is quite slow (150+ seconds / step), because GRPO generate multiple long candicate completions. Still trying to optimize it.
2. Can modify the deepspeed settings (e.g. disable CPU offload), change batch size, etc. to balance VRAM usage and training speed. Here is a guide for deepspeed: https://huggingface.co/docs/transformers/en/deepspeed