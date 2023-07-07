from flask import Flask, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from POST request
    data = request.get_json(force=True)
    text = data['text']
    
    # Encode a text inputs
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Generate text until the output length (which includes the context length) reaches 50
    output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

    # Decode the generated text
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {'generated_text': text}

if __name__ == '__main__':
    app.run(port=5000, debug=False)
