import torch
import pandas as pd
import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import evaluate
import os

# Download mandatory resources for METEOR
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# --- CONFIGURATION ---
# Path to your finetuned model directory (containing safetensors/bin and config)
MODEL_PATH = "/home/coild/NLLB/Output/Finetuning/Tam-Tel/v1"
SOURCE_LANG = "tam_Taml"
TARGET_LANG = "tel_Telu"

# Paths to your 2000 benchmark sentences already on the remote system
# Verify these paths are correct!
TEST_SOURCE_FILE = "/home/coild/NLLB/TestData/Tamil.txt"
TEST_TARGET_FILE = "/home/coild/NLLB/TestData/Telugu.txt"

# Output Filenames
OUTPUT_EXCEL = "/home/coild/NLLB/Output/Testing/Tam-Tel/V4/finalbenchmark_results_tam-tel.xlsx"
OUTPUT_TEXT = "/home/coild/NLLB/Output/Testing/Tam-Tel/V4/finalbenchmark_results_tam-tel.txt"
SCORES_FILE = "/home/coild/NLLB/Output/Testing/Tam-Tel/V4/final_score_report.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. LOAD MODEL & TOKENIZER ---
print(f"Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, src_lang=SOURCE_LANG, tgt_lang=TARGET_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

# --- 2. LOAD METRICS ---
print("Initializing metrics...")
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
ter_metric = evaluate.load("ter")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")

# --- 3. LOAD DATA ---
with open(TEST_SOURCE_FILE, 'r', encoding='utf-8') as f:
    sources = [line.strip() for line in f]
with open(TEST_TARGET_FILE, 'r', encoding='utf-8') as f:
    references = [line.strip() for line in f]

# --- 4. INFERENCE (TRANSLATION) ---
print(f"Translating {len(sources)} sentences...")
hypotheses = []
batch_size = 4  # Safe for 12GB VRAM

model.eval()
for i in tqdm(range(0, len(sources), batch_size)):
    batch_text = sources[i:i + batch_size]
    inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TARGET_LANG),
            max_length=256,
            num_beams=5
        )

    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    hypotheses.extend(decoded_preds)

# --- 5. CALCULATE SCORES ---
print("Calculating all metrics...")
formatted_refs = [[ref] for ref in references]

bleu_res = bleu_metric.compute(predictions=hypotheses, references=formatted_refs)
chrf_res = chrf_metric.compute(predictions=hypotheses, references=formatted_refs)
ter_res = ter_metric.compute(predictions=hypotheses, references=formatted_refs)
meteor_res = meteor_metric.compute(predictions=hypotheses, references=formatted_refs)
rouge_res = rouge_metric.compute(predictions=hypotheses, references=references)

# --- 6. SAVE SCORE REPORT ---
score_text = f"""--- BENCHMARK SCORE REPORT ---
SacreBLEU: {bleu_res['score']:.4f}
chrF++:    {chrf_res['score']:.4f}
TER:       {ter_res['score']:.4f} (Lower is better)
METEOR:    {meteor_res['meteor']:.4f}
ROUGE-L:   {rouge_res['rougeL']:.4f}
"""

print(score_text)
with open(SCORES_FILE, "w", encoding="utf-8") as f:
    f.write(score_text)

# --- 7. SAVE COMPARISON (EXCEL & TEXT) ---
df = pd.DataFrame({
    "Source (Tamil)": sources,
    "Benchmark (Telugu)": references,
    "Model Output": hypotheses
})

# Save to Excel
df.to_excel(OUTPUT_EXCEL, index=False)

# Save to Text
with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
    for src, ref, hyp in zip(sources, references, hypotheses):
        f.write(f"SRC: {src}\nREF: {ref}\nOUT: {hyp}\n{'-'*40}\n")

print(f"\nDone! Files generated:\n- {SCORES_FILE}\n- {OUTPUT_EXCEL}\n- {OUTPUT_TEXT}")
