Healthcare Chatbot: AI-Powered Medical Question Answering System

 Project Overview
 
Healthcare Chatbot is an intelligent conversational AI system designed to answer medical questions using a fine-tuned T5 Transformer model. The chatbot is trained on the ChatDoctor-HealthCareMagic-100k dataset, containing over 100,000 real doctor-patient conversations from online medical forums.
Purpose & Domain Alignment
This chatbot addresses the need for accessible preliminary medical information and symptom guidance. It demonstrates practical NLP applications in the healthcare domain by:

Processing medical questions in natural language
Generating contextually relevant medical responses
Providing immediate healthcare information while encouraging professional consultation
Supporting patient education and symptom awareness

Key Features
- Fine-tuned T5-small model for medical Q&A
- Comprehensive hyperparameter tuning with 6+ experiments
- Multiple evaluation metrics (BLEU, accuracy, loss)
- Clean Flask REST API backend
- Modern responsive web interface
- Input validation and error handling
- Real-time response generation

- 
Installation & Setup
Prerequisites

Python 
TensorFlow 
PyTorch  (for Hugging Face transformers)

Step 1: Clone Repository
git clone 
cd Doctor-Chatbot
Step 2: Create Virtual Environment
python -m venv venv
source venv/bin/activate 
Step 3: Install Dependencies
pip install -r requirements.txt
Step 4: Download Pre-trained Model
The model is included in healthcare_chatbot_model/

Running the Application
Backend (Flask API)
python app.py
The API will start on http://localhost:5000
Available Endpoints:

GET / - Web interface
POST /chat - Send query and get response
GET /health - Health check
GET /info - Bot information

hosted link:
https://doctor-chatbox.onrender.com
