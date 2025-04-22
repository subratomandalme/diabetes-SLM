from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Initialize app
app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(r"B:\code\basic-slm-diabetes\diabetes\basic-slm-medical-model")
model = AutoModelForSeq2SeqLM.from_pretrained(r"B:\code\basic-slm-diabetes\diabetes\basic-slm-medical-model")

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Serve Frontend
@app.route('/')
def home():
    return render_template('index.html')

# Backend API for answering questions
@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Get question from the request
        data = request.get_json()
        question = data.get('question', '')

        # Check if question is provided
        if not question.strip():
            return jsonify({'answer': 'Please provide a valid question.'}), 400

        # Create the prompt based on the question
        prompt = (
            "You are a knowledgeable medical assistant specializing in diabetes. "
            f"Please provide a detailed, coherent, and accurate response to the question below:\n{question}"
        )

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        print(f"Inputs: {inputs}")  # Debugging log to check the tokenized input

        # Generate the output using the model
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=512,  # Increased max length
            num_beams=6,
            no_repeat_ngram_size=4,
            length_penalty=1.3,
            early_stopping=True,
            temperature=0.9,
            top_p=0.95
        )

        print(f"Generated Outputs: {outputs}")  # Debugging log to check model output

        # Check if the model generated an output
        if outputs is None or len(outputs) == 0:
            return jsonify({'answer': 'Error: Model did not generate a valid response.'}), 500

        # Decode the output into a readable answer
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the generated answer
        return jsonify({'answer': answer})

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging log for exceptions
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
