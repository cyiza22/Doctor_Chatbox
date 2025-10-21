"""
Convert TensorFlow model to PyTorch and push to Hugging Face
Run this locally before deploying
"""

from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Load your TF model
print("Loading TensorFlow model...")
tf_model = TFAutoModelForSeq2SeqLM.from_pretrained("healthcare_chatbot_model")
tokenizer = AutoTokenizer.from_pretrained("healthcare_chatbot_model")

# Convert to PyTorch
print("Converting to PyTorch...")
pytorch_model = AutoModelForSeq2SeqLM.from_pretrained(
    "healthcare_chatbot_model",
    from_tf=True
)

# Save locally
print("Saving PyTorch model locally...")
pytorch_model.save_pretrained("healthcare_chatbot_model_pytorch")
tokenizer.save_pretrained("healthcare_chatbot_model_pytorch")

# Push to Hugging Face (requires login)
print("\nPushing to Hugging Face...")
print("Make sure you're logged in: huggingface-cli login")

try:
    pytorch_model.push_to_hub("Henriette22/healthcare-chatbot-t5-pytorch")
    tokenizer.push_to_hub("Henriette22/healthcare-chatbot-t5-pytorch")
    print(" Successfully pushed to Hugging Face!")
    print("Update your MODEL_PATH to: Henriette22/healthcare-chatbot-t5-pytorch")
except Exception as e:
    print(f"Error pushing to hub: {e}")
    print("You can manually upload from 'healthcare_chatbot_model_pytorch' folder")
