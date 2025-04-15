import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
from peft import LoraConfig
import json
from html.parser import HTMLParser

from reward import html_quality_reward

# Load local JSON dataset for HTML generation
def load_local_dataset(json_path):
    """
    Load a local JSON dataset in the format:
    [
      { "prompt": "...", "content": "..." },
      { "prompt": "...", "content": "..." },
      ...
    ]
    
    Returns a Dataset object compatible with GRPO training
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract prompts and completions
    prompts = [item["prompt"] for item in data]
    completions = [item["completion"] for item in data]
    
    # Create a dictionary in the format expected by Dataset.from_dict
    dataset_dict = {
        "prompt": prompts,  # The prompts
        "completion": completions,  # The completions (HTML content)
    }
    
    # Create and return the Dataset
    return Dataset.from_dict(dataset_dict)

# Path to your JSON file - update this to point to your actual JSON file
json_dataset_path = "/ephemeral/grpo-data/dojo_sft_js_grpo-format.json"  

# Load the dataset
dataset = load_local_dataset(json_dataset_path)

# You can also limit the dataset size for faster experimentation
# dataset = dataset.select(range(min(5, len(dataset))))


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"],  # update these target modules as appropriate
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Configure the training
training_args = GRPOConfig(
    output_dir="/ephemeral/checkpoints/Qwen2.5-Coder-7B-GRPO", # NOTE: Directory to save the model checkpoints. Change if necessary.
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_completion_length=1024,
    learning_rate=1e-5,
    logging_steps=10,
    num_generations=4,
    save_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    beta=0.02,
    bf16=True,  # Use mixed precision training
    logging_dir="Log-Qwen2.5-Coder-7B-GRPO",
    deepspeed="deepspeed_config_grpo.json",
    label_names=["completion"],
)

# Initialize the tokenizer
model_name_or_path = "Qwen/Qwen2.5-Coder-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Initialize the GRPO trainer
trainer = GRPOTrainer(
    model=model_name_or_path,
    # tokenizer=tokenizer,
    reward_funcs=html_quality_reward,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    # max_length=1024,
    # max_prompt_length=256,
    # input_field="input",  # The field containing the prompts in our dataset
    # completion_field="completion",  # The field containing the completions in our dataset
)

# Define a callback to pass prompts to the reward function
# class PromptAwareRewardCallback:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.prompt_map = {i: item["input"] for i, item in enumerate(dataset)}
        
#     def on_evaluate_completion(self, args, kwargs, batch_indices, **callback_kwargs):
#         # Add prompts to the kwargs that will be passed to the reward function
#         batch_prompts = [self.prompt_map[int(idx)] for idx in batch_indices]
#         kwargs["prompts"] = batch_prompts
#         return args, kwargs

# # Add the callback to the trainer
# trainer.add_callback(PromptAwareRewardCallback(dataset))

# Start training
trainer.train()

