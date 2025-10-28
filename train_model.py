# train_model.py
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import torch
import json

# ----------------------------
# Load your dataset
# ----------------------------
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = Dataset.from_dict({
        "prompt": [entry["prompt"] for entry in data],
        "response": [entry["response"] for entry in data]
    })
    return dataset

# ----------------------------
# Prompt formatting
# ----------------------------
chat_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    EOS_TOKEN = tokenizer.eos_token
    texts = [
        chat_prompt.format("", p, r) + EOS_TOKEN
        for p, r in zip(examples["prompt"], examples["response"])
    ]
    return {"text": texts}

# ----------------------------
# Load model
# ----------------------------
max_seq_length = 2056
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# ----------------------------
# Prepare LoRA (fine-tuning)
# ----------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ----------------------------
# Train the model
# ----------------------------
dataset = load_dataset("arabic_dataset.json")
dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        max_steps=60,
        learning_rate=2e-4,
        output_dir="outputs",
        logging_steps=1,
        optim="adamw_8bit",
        fp16=True,
    ),
)

trainer.train()

# ----------------------------
# Save model & tokenizer
# ----------------------------
model.save_pretrained("fine_tuned_llama3_arabic")
tokenizer.save_pretrained("fine_tuned_llama3_arabic")

print("  Model saved in fine_tuned_llama3_arabic/")
