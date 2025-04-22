
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from pathlib import Path
import os


dir_path = Path(__file__).parent.resolve()
os.chdir(dir_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_file = dir_path / "medical_qa.csv"
if not data_file.exists():
    raise FileNotFoundError(f"CSV not found: {data_file}")
dataset = load_dataset("csv", data_files={"train": str(data_file)}, split="train")
print(f"Loaded {len(dataset)} samples")


model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)


def preprocess_function(examples):
    inputs = [f"question: {q}" for q in examples["question"]]
    targets = examples["answer"]
    model_inputs = tokenizer(inputs, truncation=True)
    labels = tokenizer(targets, truncation=True).input_ids
    
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in seq]
        for seq in labels
    ]
    return model_inputs

tokenized = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


output_dir = dir_path / "basic-slm-medical-model"
training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to=None
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer
)
trainer.train()


model.save_pretrained(str(output_dir))
tokenizer.save_pretrained(str(output_dir))
print("Model saved to", output_dir)



"""
Zero-shot diabetes Q&A using a larger instruction-tuned FLAN-T5 model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import os


script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def answer_question(
    question: str,
    max_len: int = 256,
    num_beams: int = 6,
    no_repeat_ngram_size: int = 4,
    length_penalty: float = 1.3
):
    prompt = (
        "You are a knowledgeable medical assistant specializing in diabetes. "
        f"Please provide a detailed, coherent, and accurate response to the question below:\n{question}"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_len,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        early_stopping=True,
        temperature=0.9,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print("FLAN-T5 Diabetes Q&A (zero-shot) with FLAN-T5-large. Type 'exit' to quit.")
    while True:
        q = input("Q: ")
        if q.strip().lower() in ["exit", "quit"]:
            break
        ans = answer_question(q)
        print(f"A: {ans}\n")

if __name__ == '__main__':
    main()