from flask import Flask, request, jsonify
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

model_name = "healthcare_chatbot_model"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("question")
    input_text = "question: " + user_input
    inputs = tokenizer(input_text, return_tensors="tf")
    output = model.generate(**inputs, max_length=80)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"response": answer})

# @app.route("/")
# def home():
#     return "Healthcare Chatbot is running!"

if __name__ == "__main__":
    app.run(debug=True)
