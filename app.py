import gradio as gr
from transformers import pipeline

# 1. Connect to your trained brain
MODEL_ID = "shashibekka/kgp-student-finance-model" 
print("Waking up the AI Advisor...")
bot_pipeline = pipeline("text-generation", model=MODEL_ID, device_map="auto")

# 2. Define the Chat Logic (Notice we handle 'history' now like a real chat app)
def predict(message, history):
    prompt = f"""Below is an instruction from a student. Write a helpful financial response.
### Instruction:
{message}
### Response:
"""
    result = bot_pipeline(prompt, max_new_tokens=150, return_full_text=False)
    return result[0]['generated_text'].strip()

# 3. Injecting Custom CSS for that Instagram DM vibe
custom_css = """
/* Make the whole app look sleek */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #fafafa;
}
/* Style the User's sent messages with an Instagram-style gradient */
.user {
    background: linear-gradient(45deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%) !important;
    color: white !important;
    border-radius: 22px 22px 4px 22px !important;
}
/* Style the AI's received messages */
.bot {
    background-color: #efefef !important;
    color: #262626 !important;
    border-radius: 22px 22px 22px 4px !important;
}
"""

# 4. Build the Chat App
demo = gr.ChatInterface(
    predict,
    title="IIT KGP Finance AI 💸✨",
    description="Drop a message to get campus budgeting advice! 🎓📊",
    theme=gr.themes.Base(),
    css=custom_css,
    examples=["I have 500 rupees left, should I go to PFC?", "How do I save on my mess bill?"],
)

# 5. Launch
if __name__ == "__main__":
    demo.launch()