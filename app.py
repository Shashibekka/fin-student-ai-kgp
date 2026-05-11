import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 1. Download the compressed GGUF model from your new repo
model_path = hf_hub_download(
    repo_id="shashibekka/kgp-phi3-finance-gguf",
    filename="phi-3-mini-4k-instruct.Q4_K_M.gguf" # <-- The new Phi-3 name!
)

# 2. Load the model directly into CPU memory (ON A STRICT DIET)
print("Waking up the AI Advisor...")
llm = Llama(
    model_path=model_path,
    n_ctx=512,        # LOWERED: Only keep ~500 words in short-term memory at a time
    n_batch=128,      # ADDED: Process the math in much smaller chunks to save RAM
    n_threads=2,      # Free HF Spaces give us 2 CPU cores
)

# 3. Define the Chat Logic (NOW WITH STREAMING & PERSONA)
def predict(message, history):
    # We inject a secret persona before the user's message to make it sound human
    prompt = f"""You are ChatKGP, a friendly, casual, and highly knowledgeable financial advisor for IIT Kharagpur students. Talk like a real human student. Be conversational, empathetic, and relatable. Avoid robotic one-liners. 

### Instruction:
{message}
### Response:
"""
    
    # We added stream=True here so it sends words one by one!
    stream = llm(prompt, max_tokens=200, stop=["<|end_of_text|>", "<|eot_id|>"], stream=True)
    
    partial_message = ""
    for chunk in stream:
        # Extract the token and add it to the message
        token = chunk['choices'][0]['text']
        partial_message += token
        yield partial_message  # 'yield' pushes it to the UI instantly!

# 4. The New UI (Sleek, Dark, Gradient Style)

# Define the sleek, dark base theme
custom_theme = gr.themes.Soft(
    primary_hue="emerald",  # Matches the green vibe
    neutral_hue="slate",
).set(
    body_background_fill="#0B0F19", # Deep dark background
    block_background_fill="#111827", # Slightly lighter panels
    block_border_width="1px",
    block_border_color="#1F2937",
    input_background_fill="#1F2937",
)

# Inject Custom CSS for the glowing gradients
custom_css = """
/* Gradient effect for the main title */
h1 {
    text-align: center;
    background: linear-gradient(90deg, #10B981, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Gradient border for the chat input box */
.contain textarea {
    border: 1px solid transparent !important;
    background: linear-gradient(#1F2937, #1F2937) padding-box, 
                linear-gradient(90deg, #10B981, #3B82F6) border-box !important;
    border-radius: 8px;
}

/* Subtle gradient background for the AI's chat bubbles */
.bot {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.05)) !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
}
"""

# Apply the theme, CSS, and Labels to your ChatInterface
demo = gr.ChatInterface(
    fn=predict, 
    title="ChatKGP", # <-- Removed the emoji
    chatbot=gr.Chatbot(label="ChatKGP", show_label=True), # <-- Replaced the "Chatbot" text
    theme=custom_theme,
    css=custom_css,
    fill_height=True # Makes the chat window stretch to fit the screen nicely
)

if __name__ == "__main__":
    demo.launch()