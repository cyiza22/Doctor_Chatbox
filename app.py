"""
Healthcare Chatbot Flask Backend - PyTorch Version (Memory Optimized)
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os
import torch

# CONFIGURATION & SETUP
app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MODEL INITIALIZATION
MODEL_PATH = "Henriette22/healthcare-chatbot-t5"

try:
    logger.info("Loading model and tokenizer from Hugging Face...")
    hf_token = os.environ.get("HF_TOKEN")

    # Load with PyTorch - more memory efficient
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_auth_token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH, 
        use_auth_token=hf_token,
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True      # Critical for memory optimization
    )
    
    # Set to eval mode to save memory
    model.eval()
    
    logger.info("✓ Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    tokenizer = None

# ROUTES
@app.route("/", methods=["GET"])
def home():
    """Serve the main HTML interface"""
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return jsonify({"error": "Failed to load interface"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Process user question and return chatbot response"""
    try:
        if model is None or tokenizer is None:
            return jsonify({
                "response": "Error: Model not loaded",
                "status": "error"
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                "response": "Error: No JSON data provided",
                "status": "error"
            }), 400
        
        user_question = data.get("question", "").strip()
        
        if not user_question:
            return jsonify({
                "response": "Please enter a medical question.",
                "status": "error"
            }), 400
        
        if len(user_question) > 500:
            return jsonify({
                "response": "Question is too long. Please keep it under 500 characters.",
                "status": "error"
            }), 400
        
        # Prepare input
        input_text = "medical question: " + user_question
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
        
        # Generate response (no gradient tracking to save memory)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        disclaimer = "\n\n⚠️ Disclaimer: This is an AI-generated response. Always consult a qualified healthcare professional for medical advice."
        
        logger.info(f"Query processed: {user_question[:50]}...")
        
        return jsonify({
            "response": response + disclaimer,
            "status": "success"
        }), 200
    
    except Exception as e:
        logger.error(f"Unexpected error in /chat: {str(e)}")
        return jsonify({
            "response": f"Error: {str(e)}",
            "status": "error"
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Check if API is running and model is loaded"""
    model_loaded = model is not None and tokenizer is not None
    return jsonify({
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded
    }), 200 if model_loaded else 503


@app.route("/info", methods=["GET"])
def get_info():
    """Get information about the chatbot"""
    return jsonify({
        "name": "Healthcare Chatbot",
        "version": "1.0.0",
        "model": "T5-small (fine-tuned on ChatDoctor dataset)",
        "description": "An AI chatbot trained to answer medical questions",
        "disclaimer": "This tool is for informational purposes only. Always consult qualified healthcare professionals.",
    }), 200


# ERROR HANDLERS
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting Healthcare Chatbot API...")
    logger.info("Available endpoints:")
    logger.info("  GET  / - Main interface")
    logger.info("  POST /chat - Chat with the bot")
    logger.info("  GET  /health - Health check")
    logger.info("  GET  /info - Bot information")
    
    app.run(debug=True, host="0.0.0.0", port=5000)
