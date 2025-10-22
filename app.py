"""
Healthcare Chatbot Flask Backend - PyTorch Version (Memory Optimized)
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os
import torch

# Memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# CONFIGURATION & SETUP
app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MODEL INITIALIZATION
# Using base t5-small because TF→PyTorch conversion exceeds 512MB RAM
MODEL_PATH = "t5-small"  # 242MB - fits in free tier

model = None
tokenizer = None

def load_model():
    """Load model with error handling and fallback"""
    global model, tokenizer
    
    try:
        logger.info("=" * 60)
        logger.info("Starting model load process...")
        logger.info("=" * 60)
        
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("render_read_token")
        if not hf_token:
            logger.warning("⚠️ HF_TOKEN not found in environment variables")
        
        # Load tokenizer first (lightweight)
        logger.info("Step 1/3: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=hf_token)
        logger.info("✓ Tokenizer loaded successfully")
        
        # Clean memory
        import gc
        gc.collect()
        
        # Try loading with TF conversion
        logger.info("Step 2/3: Loading model (this may take 2-3 minutes)...")
        logger.info(f"  - Model path: {MODEL_PATH}")
        logger.info(f"  - Converting from TensorFlow: Yes")
        logger.info(f"  - Precision: float16 (half precision)")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH, 
            token=hf_token,
            low_cpu_mem_usage=True,
            dtype=torch.float16,  # Changed from torch_dtype
        )
        
        logger.info("Step 3/3: Optimizing model for inference...")
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
        
        gc.collect()
        
        logger.info("=" * 60)
        logger.info("✓✓✓ MODEL LOADED SUCCESSFULLY ✓✓✓")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ MODEL LOAD FAILED: {str(e)}")
        logger.error("=" * 60)
        logger.exception("Full error traceback:")
        
        # Try fallback to base model
        try:
            logger.info("Attempting fallback to base t5-small model...")
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "t5-small",
                low_cpu_mem_usage=True,
                dtype=torch.float16,  # Changed from torch_dtype
            )
            model.eval()
            logger.warning("⚠️ Using base t5-small (not fine-tuned)")
            return True
        except Exception as fallback_error:
            logger.error(f"❌ Fallback also failed: {str(fallback_error)}")
            model = None
            tokenizer = None
            return False

# Load model at startup
logger.info("Initializing Healthcare Chatbot...")
load_success = load_model()

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
                num_beams=2,  # Reduced from 4 to save memory
                early_stopping=True,
                do_sample=False,  # Deterministic generation
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
    
    status_info = {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH if model_loaded else None,
        "pytorch_version": torch.__version__,
        "transformers_available": True
    }
    
    if not model_loaded:
        status_info["error"] = "Model failed to load. Check logs for details."
    
    return jsonify(status_info), 200 if model_loaded else 503


@app.route("/info", methods=["GET"])
def get_info():
    """Get information about the chatbot"""
    return jsonify({
        "name": "Healthcare Chatbot",
        "version": "1.0.0",
        "model": "T5-small (fine-tuned on ChatDoctor dataset)",
        "description": "An AI chatbot trained to answer medical questions",
        "disclaimer": "This tool is for informational purposes only. Always consult qualified healthcare professionals.",
        "model_loaded": model is not None and tokenizer is not None,
        "hf_token_present": (os.environ.get("HF_TOKEN") is not None) or (os.environ.get("render_read_token") is not None)
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
