# Arabic-Chatbot

This project demonstrates a **fine-tuned LLaMA 3 model** trained on a custom Arabic dataset for conversational tasks. The main focus is **evaluating the chatbot’s responses**, rather than developing a sophisticated interface.

We provide a simple Streamlit interface to interact with the model, but the core purpose is to **showcase the quality of the generated answers** in Arabic.

---

## 🔹 Project Highlights

- Fine-tuned **LLaMA 3 (8B, 4-bit)** using **LoRA** for efficient parameter tuning.
- Dataset is in **Arabic** (`arabic_dataset.json`) with paired prompts and responses.
- Focus is on **model performance and outputs**, not the front-end interface.
- Model can generate meaningful and context-aware responses in Arabic.

---

## 🛠️ How It Works

1. **Dataset Preparation**:  
   JSON file containing `"prompt"` and `"response"` fields. Prompts are questions or instructions, and responses are the expected answers.

2. **Fine-Tuning**:  
   Using `unsloth`, `trl`, and `peft`, the pre-trained LLaMA 3 model is fine-tuned with LoRA adapters.  
   Training parameters:

   - Batch size: 2
   - Gradient accumulation steps: 16
   - Max steps: 60
   - Learning rate: 2e-4

3. **Inference / Chatbot**:  
   After fine-tuning, the model can generate responses to new Arabic prompts.

   - Input: a user question in Arabic
   - Output: chatbot response

4. **Streamlit Interface (Optional)**:
   - Simple web app for inputting questions and viewing responses.
   - Only used to demonstrate results, not the UI design.

---

## 📷 Example Result
<img width="998" height="315" alt="image" src="https://github.com/user-attachments/assets/9d329ec0-63ba-40d9-836e-3a44a5bf4acb" />

---

