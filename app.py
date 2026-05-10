import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 1. Download the compressed GGUF model from your new repo
print("Downloading GGUF model...")
model_path = hf_hub_download(
    repo_id="shashibekka/kgp-finance-gguf", 
    filename="unsloth.Q4_K_M.gguf"
)

# 2. Load the model directly into CPU memory (ON A STRICT DIET)
print("Waking up the AI Advisor...")
llm = Llama(
    model_path=model_path,
    n_ctx=512,        # LOWERED: Only keep ~500 words in short-term memory at a time
    n_batch=128,      # ADDED: Process the math in much smaller chunks to save RAM
    n_threads=2,      # Free HF Spaces give us 2 CPU cores
)

# 3. Define the Chat Logic (NOW WITH STREAMING!)
def predict(message, history):
    prompt = f"""Below is an instruction from a student. Write a helpful financial response.
### Instruction:
{message}
### Response:
"""
    
    # We added stream=True here so it sends words one by one!
    stream = llm(prompt, max_tokens=150, stop=["<|end_of_text|>", "<|eot_id|>"], stream=True)
    
    partial_message = ""
    for chunk in stream:
        # Extract the token and add it to the message
        token = chunk['choices'][0]['text']
        partial_message += token
        yield partial_message  # 'yield' pushes it to the UI instantly!

# 4. The UI (Same Instagram style as before!)
custom_css = """
.gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #fafafa; }
.user { background: linear-gradient(45deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%) !important; color: white !important; border-radius: 22px 22px 4px 22px !important; }
.bot { background-color: #efefef !important; color: #262626 !important; border-radius: 22px 22px 22px 4px !important; }
"""

demo = gr.ChatInterface(
    predict,
    title="IIT KGP Finance AI 💸✨",
    description="Drop a message to get campus budgeting advice! 🎓📊",
    theme=gr.themes.Base(),
    css=custom_css,
    examples=["I have 500 rupees left, should I go to PFC?", "How do I save on my mess bill?"],
)

if __name__ == "__main__":
    demo.launch()