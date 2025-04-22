from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Initialize app
app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./basic-slm-medical-model")
model = AutoModelForSeq2SeqLM.from_pretrained("./basic-slm-medical-model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Serve Frontend
@app.route('/')
def home():
    return render_template('index.html')

# Backend API
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    if not question.strip():
        return jsonify({'answer': 'Please provide a valid question.'}), 400

    prompt = (
        "You are a knowledgeable medical assistant specializing in diabetes. "
        f"Please provide a detailed, coherent, and accurate response to the question below:\n{question}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=256,
        num_beams=6,
        no_repeat_ngram_size=4,
        length_penalty=1.3,
        early_stopping=True,
        temperature=0.9,
        top_p=0.95
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'answer': answer})

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
