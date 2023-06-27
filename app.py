import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder="/",
)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")


def create_embedding(input_text):
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
    print(output_text)
    return output_text


instructor_model_embeddings = gr.Interface(
    fn=create_embedding,
    inputs=[
        gr.inputs.Textbox(label="Input Text"),
    ],
    outputs=gr.inputs.Textbox(label="Generated Text"),
    title="Falcon-7B Instruct",
).launch()
