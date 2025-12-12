# data_assistant.py
import os
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load API Token
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env")

client = InferenceClient(
    model="deepseek-ai/DeepSeek-V3.1",  # replace with other model if needed
    token=HF_TOKEN
)

def summarize_dataset(df: pd.DataFrame) -> str:
    """Create a dataset summary including numeric stats + categorical info."""
    summary = []

    # Basic info
    summary.append(f"Shape: {df.shape}")
    summary.append(f"Columns: {list(df.columns)}")

    # Numeric stats
    numeric = df.describe(include="number").to_string()
    summary.append(f"Numeric summary:\n{numeric}")

    # Categorical stats
    categorical = df.describe(include="object").to_string()
    summary.append(f"Categorical summary:\n{categorical}")

    return "\n\n".join(summary)


def ask_model(question: str, df: pd.DataFrame) -> str:
    """
    Ask model with enriched dataset context.
    """
    context = summarize_dataset(df)

    messages=[
    {"role": "system", "content": "You are a witty, sarcastic but still helpful data analyst. You always explain answers with a touch of humor and sarcasm."},
    {"role": "user", "content": question}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=400
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error from model: {str(e)}"
