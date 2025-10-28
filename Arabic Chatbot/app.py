# app.py
import streamlit as st
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch

st.set_page_config(page_title="Arabic Chatbot", page_icon="💬", layout="centered")

@st.cache_resource
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="fine_tuned_llama3_arabic",
        max_seq_length=2056,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

st.title("🤖 Arabic Chatbot (Fine-Tuned LLaMA 3)")

user_input = st.text_area("💬 اكتب سؤالك هنا:")

if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        chat_prompt = f"### Instruction:\n\n### Input:\n{user_input}\n\n### Response:\n"
        inputs = tokenizer([chat_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        decoded_output = tokenizer.batch_decode(outputs)[0]
        response = decoded_output.split("### Response:")[-1].split("<|end_of_text|>")[0].strip()
        st.success(response)
