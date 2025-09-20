from flask import Flask, render_template, request
import pickle
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Load Model and Vectorizer ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    print("Model files not found! Please run model.py first to train and save them.")
    model = None
    vectorizer = None
# --- END ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model and vectorizer and request.method == 'POST':
        # 1. Get the email text from the form
        email_text = request.form['email_text']
        
        # 2. Transform the input text using the loaded vectorizer
        email_vec = vectorizer.transform([email_text])
            
        # 3. Make a prediction using the loaded model
        prediction = model.predict(email_vec)
        
        # 4. Determine the result string and a CSS class for styling
        if prediction[0] == 1:
            result = 'Spam'
            result_class = 'spam'
        else:
            result = 'Not Spam'
            result_class = 'not-spam'
        
        # 5. Render the home page again with the prediction result and class
        return render_template('index.html', prediction=result, result_class=result_class)
    else:
        return render_template('index.html', prediction="Error: Model not loaded.", result_class="error")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
