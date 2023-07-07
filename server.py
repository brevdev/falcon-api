import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")


def generate_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove Prompt Echo from Generated Text
    cleaned_output_text = output_text.replace(input_text, "")
    return cleaned_output_text


@app.route('/predictions', methods=['POST'])
def predict():
    print(request)
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400
    input_text = data.get('input_text', '')
    if not input_text:
        return jsonify({'error': 'input_text is required'}), 400
    prediction = generate_text(input_text)
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(port=4999, debug=True)
