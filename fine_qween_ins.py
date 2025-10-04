import json
import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling,BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
load_dotenv()
print("🔹 Step 1: Loading dataset...")
with open("conversation_dataset_ShirayukiV3.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"✅ Dataset loaded. Total samples: {len(data)}")

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(data)
print("✅ Converted to HuggingFace Dataset")

# Rename columns
dataset = dataset.rename_columns({"guy": "input", "girl": "output"})
print("✅ Renamed columns:", dataset.column_names)

# Merge into one text field
def format_examples(example):
    return {
        "text": f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_examples)
print("✅ Formatted dataset with instruction-response pairs")
print("Sample formatted text:\n", dataset[0]["text"])

# -----------------
# Load Model + Tokenizer
# -----------------
print("🔹 Step 2: Loading model and tokenizer...")
model_name = "Qwen/Qwen3-1.7B"  # change if needed
token = os.getenv("HF_TOKEN")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,          # or load_in_8bit=True
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name,token=token,trust_remote_code=True,
    timeout=300)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model.tie_weights()
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],  # tweak depending on architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✅ Model loaded")

# -----------------
# Tokenize Dataset
# -----------------
print("🔹 Step 3: Tokenizing dataset...")

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
print("✅ Dataset tokenized")
print("Sample tokenized input_ids:", tokenized_dataset[0]["input_ids"][:20])

# -----------------
# Training Arguments
# -----------------

print("🔹 Step 4: Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./outputs",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="adamw_torch",
)
print("✅ Training arguments ready")

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
print("✅ Data collator created")

# -----------------
# Trainer
# -----------------
print("🔹 Step 5: Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("✅ Trainer initialized")

print("🚀 Training started ...")
trainer.train()
print("🏁 Training completed.")

# Save the PEFT adapter (LoRA weights only)
trainer.save_model("./lora_adapter_qween")

# Save tokenizer (important so inference works later)
tokenizer.save_pretrained("./lora_adapter_qween")

print("✅ LoRA adapter saved at ./lora_adapter_qween")

