from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load the pretrained model and tokenizer from local files
model_path = "kristine-yolo/BERTwatch"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Define class labels
class_names = [
    'Abuse',
    'Anxiety',
    'Death',
    'Depression',
    'Domestic Violence',
    'Dysmorphia',
    'Eating Disorder',
    'PTSD',
    'Suicide'
]

# Serve web pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/classifier')
def classifier():
    return render_template('classifier.html')

# API endpoint for classifying text
@app.route('/api/classify', methods=['POST'])
def classify_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)

    # Get top-3 predictions
    topk_probs, topk_indices = torch.topk(probs, k=3, dim=1)

    results = [
        {
            "label": class_names[idx],
            "probability": float(prob) 
        }
        for idx, prob in zip(topk_indices[0].cpu().numpy(), topk_probs[0].cpu().numpy())
    ]

    return jsonify({"predictions": results})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
