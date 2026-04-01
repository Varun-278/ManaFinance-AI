import gradio as gr
from groq import Groq

# Configuration
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID = "llama-3.1-8b-instant"

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


# Translation Function
def translate(text: str, direction: str) -> str:
    """
    Translate between English and Telugu using Groq-hosted LLaMA-3.1-8B-Instant.
    """
    text = text.strip()
    if not text:
        return "Please enter some text."

    if direction == "English → Telugu":
        prompt = f"Translate the following English text into fluent Telugu:\n\n{text}\n\nMake it natural and grammatically correct."
    else:
        prompt = f"Translate the following Telugu text into clear and fluent English:\n\n{text}\n\nEnsure proper grammar and readability."

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,   # allow longer translations
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Translation error: {str(e)}"


# Gradio Interface
demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(
            label="Enter Text",
            placeholder="Type or paste your text here...",
            lines=4
        ),
        gr.Dropdown(
            ["English → Telugu", "Telugu → English"],
            label="Translation Direction"
        ),
    ],
    outputs=gr.Textbox(
        label="Translated Output",
        lines=8,  #  makes the output box bigger
        placeholder="Your translated text will appear here..."
    ),
    title="🌐 English ↔ Telugu Translator (Groq LLaMA-3.1-8B-Instant)",
    description="Instant bilingual translations powered by Groq API. Uses the LLaMA-3.1-8B model for accurate and natural results.",
    theme="soft",
    examples=[
        ["The Reserve Bank of India kept the repo rate unchanged at 6.50 percent.", "English → Telugu"],
        ["రిపో రేటు 6.50 శాతం వద్ద యథాతథంగా ఉంచబడింది.", "Telugu → English"],
    ]
)


#  Launch App
if __name__ == "__main__":
    demo.launch()
