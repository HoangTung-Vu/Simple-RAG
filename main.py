from flask import Flask, render_template, request, jsonify
from rag import run_rag_app
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    try:
        return render_template('index.html')  # Remove '/templates/' from path
    except Exception as e:
        app.logger.error(f"Home route error: {str(e)}")
        return str(e), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        app.logger.debug(f"Received request data: {request.json}")
        question = request.json.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        app.logger.debug(f"Processing question: {question}")
        response = run_rag_app(question)
        return jsonify({'answer': response})
    except Exception as e:
        app.logger.error(f"Ask route error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)