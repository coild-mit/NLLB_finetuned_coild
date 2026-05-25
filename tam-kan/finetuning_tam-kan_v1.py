import os
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

# 1. Configuration & Flexibility
MODEL_ID = "facebook/nllb-200-distilled-600M"
# Note: It is safer to use hf login, but keeping token variable as requested
HF_TOKEN = "HF_TOKEN"

# Switch these variables to swap source and target
SOURCE_LANG = "tam_Taml"  # Tamil
TARGET_LANG = "kan_Knda"  # Kannada

# Updated paths for the remote system
SOURCE_FILE = "/home/coild/NLLB/TrainingData/tam-kan/Tamil.txt"
TARGET_FILE = "/home/coild/NLLB/TrainingData/tam-kan/Kannada.txt"
OUTPUT_DIR = "/home/coild/NLLB/Output/Finetuning"

# 2. Data Loading & Preprocessing
def load_custom_data(src_file, tgt_file):
    if not os.path.exists(src_file) or not os.path.exists(tgt_file):
        raise FileNotFoundError(f"Check your paths: {src_file} or {tgt_file} not found.")
        
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f]
    
    data_list = [{"translation": {SOURCE_LANG: s, TARGET_LANG: t}} for s, t in zip(src_lines, tgt_lines)]
    return Dataset.from_list(data_list)

# Load training data
raw_train_dataset = load_custom_data(SOURCE_FILE, TARGET_FILE)

# Load IN22-Gen for validation using the modern 'default' parquet format
print("Loading IN22-Gen validation set...")
full_in22 = load_dataset("ai4bharat/IN22-Gen", "default", split="test")

# 1. Dynamically find the right column names
col_names = full_in22.column_names
tam_col = [c for c in col_names if "tam" in c.lower()][0]
kan_col = [c for c in col_names if "kan" in c.lower()][0]

print(f"Detected columns - Tamil: {tam_col}, Kannada: {kan_col}")

# 2. Updated function using the detected keys
def format_in22(example):
    return {
        "translation": {
            SOURCE_LANG: example[tam_col],
            TARGET_LANG: example[kan_col]
        }
    }

# 3. Map the dataset
validation_dataset = full_in22.map(format_in22, remove_columns=full_in22.column_names)

# Combine into DatasetDict
dataset = DatasetDict({
    "train": raw_train_dataset,
    "validation": validation_dataset
})

# 3. Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    token=HF_TOKEN, 
    src_lang=SOURCE_LANG, 
    tgt_lang=TARGET_LANG
)
# Force the use of safetensors to bypass the CVE-2025-32434 security check
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID, 
    token=HF_TOKEN, 
    use_safetensors=True
)

max_length = 128

def preprocess_function(examples):
    inputs = [ex[SOURCE_LANG] for ex in examples["translation"]]
    targets = [ex[TARGET_LANG] for ex in examples["translation"]]
    
    # Standard way for modern Seq2Seq tokenizers: 
    # Use 'text' for source and 'text_target' for the labels
    model_inputs = tokenizer(
        inputs, 
        text_target=targets, 
        max_length=max_length, 
        truncation=True
    )

    return model_inputs

# Map the preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=2e-5,
    # --- MEMORY OPTIMIZATION START ---
    per_device_train_batch_size=1,       # Reduce from 4 to 1
    gradient_accumulation_steps=4,      # Accumulate 4 steps to keep effective batch size at 4
    optim="adamw_bnb_8bit",             # Use 8-bit AdamW (saves ~50% optimizer memory)
    gradient_checkpointing=True,        # Trades compute for memory
    # --- MEMORY OPTIMIZATION END ---
    per_device_eval_batch_size=1,       # Reduce eval batch size as well
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True, 
    push_to_hub=False,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 5. Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer, # Changed from tokenizer=tokenizer
    data_collator=data_collator,
)

# 6. Start Training
print("Starting training...")
trainer.train()

# 7. Save the model
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
