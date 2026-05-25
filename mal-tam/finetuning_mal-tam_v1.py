import os
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

# 1. Configuration
MODEL_ID = "facebook/nllb-200-distilled-600M"
HF_TOKEN = "HF_TOKEN"

# NLLB Language Codes: Tamil -> Malayalam
SOURCE_LANG = "mal_Mlym"
TARGET_LANG = "tam_Taml"

# Updated Paths for Tamil-Malayalam pair
SOURCE_FILE = "/home/coild/NLLB/TrainingData/tam-mal/Malayalam.txt"
TARGET_FILE = "/home/coild/NLLB/TrainingData/tam-mal/Tamil.txt"
OUTPUT_DIR = "/home/coild/NLLB/Output/Finetuning/Mal-Tam"

# 2. Data Loading & Dynamic Preprocessing
def load_custom_data(src_file, tgt_file):
    if not os.path.exists(src_file) or not os.path.exists(tgt_file):
        raise FileNotFoundError(f"Source or Target file missing at: {src_file}")

    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f]
    
    # Post-processing: Ensure parallel alignment
    min_len = min(len(src_lines), len(tgt_lines))
    data_list = [{"translation": {SOURCE_LANG: src_lines[i], TARGET_LANG: tgt_lines[i]}} for i in range(min_len)]
    return Dataset.from_list(data_list)

# Load Training Data
raw_train_dataset = load_custom_data(SOURCE_FILE, TARGET_FILE)

# Load Validation Data (IN22-Gen)
print("Loading IN22-Gen validation set...")
full_in22 = load_dataset("ai4bharat/IN22-Gen", "default", split="test")

# --- DYNAMIC COLUMN DETECTION (Tamil & Malayalam) ---
col_names = full_in22.column_names
try:
    # Look for columns containing 'tam' (Tamil) and 'mal' (Malayalam)
    tam_col = [c for c in col_names if "tam" in c.lower() and "sentence" in c.lower()][0]
    mal_col = [c for c in col_names if "mal" in c.lower() and "sentence" in c.lower()][0]
except IndexError:
    # Fallback to language codes if naming convention differs
    tam_col = [c for c in col_names if "tam" in c.lower()][0]
    mal_col = [c for c in col_names if "mal" in c.lower()][0]

print(f"Verified Validation Columns: Tamil='{tam_col}', Malayalam='{mal_col}'")

def format_in22(example):
    return {
        "translation": {
            SOURCE_LANG: example[mal_col],
            TARGET_LANG: example[tam_col]
        }
    }

# Process validation set and remove original raw columns to save memory
validation_dataset = full_in22.map(format_in22, remove_columns=full_in22.column_names)

# Combine into DatasetDict
dataset = DatasetDict({
    "train": raw_train_dataset,
    "validation": validation_dataset
})

# 3. Model & Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    token=HF_TOKEN, 
    src_lang=SOURCE_LANG, 
    tgt_lang=TARGET_LANG
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID, 
    token=HF_TOKEN, 
    use_safetensors=True
)

max_length = 128

def preprocess_function(examples):
    inputs = [ex[SOURCE_LANG] for ex in examples["translation"]]
    targets = [ex[TARGET_LANG] for ex in examples["translation"]]
    
    model_inputs = tokenizer(
        inputs, 
        text_target=targets, 
        max_length=max_length, 
        truncation=True
    )
    return model_inputs

# Map preprocessing and clean up columns
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. Training Arguments (20 Epochs, Patience logic enabled)
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=20,

    # Early Stopping Requirements
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,

    # Memory Optimization for Workstation (Ubuntu/Precision 3680)
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_bnb_8bit",
    gradient_checkpointing=True,
    per_device_eval_batch_size=1,

    weight_decay=0.01,
    save_total_limit=2, # Keeps only best + last to save disk space
    predict_with_generate=True,
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 5. Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    # Patience set to 4 epochs
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

# 6. Start Fine-tuning
print(f"Starting Malayalam-to-Tamil fine-tuning...")
trainer.train()

# 7. Save Final (Best) Model
trainer.save_model(OUTPUT_DIR)
print(f"Training finished. Best Malayalam-to-Tamil model saved to {OUTPUT_DIR}")
