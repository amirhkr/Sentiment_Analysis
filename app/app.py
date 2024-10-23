from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# Define the SentimentClassifier class
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # BERT returns a tuple: (last_hidden_state, pooled_output)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # Extract the pooled_output (i.e., the [CLS] token representation)

        output = self.drop(pooled_output)  # Apply dropout to the pooled output
        return self.out(output)

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Ensure the model is loaded correctly
try:
    model = torch.load('../model/model.pth')
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model is not None:
    model = model.to(device)

# Prediction function
def predict_sentiment(review_text):
    inputs = tokenizer.encode_plus(
        review_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

    return "positive" if prediction.item() == 1 else "negative"

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# POST route for sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'review' not in request.json:
        return jsonify({'error': 'Invalid input'}), 400
    
    review = request.json['review']
    sentiment = predict_sentiment(review)
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
