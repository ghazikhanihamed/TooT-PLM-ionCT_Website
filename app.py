from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit_sequence', methods=['POST'])
def submit_sequence():
    sequence = request.form['proteinSequence']
    # Process the sequence
    # For now, just return it as a response
    return jsonify({'sequence': sequence})

@app.route('/submit_file', methods=['POST'])
def submit_file():
    if 'fastaFile' in request.files:
        fasta_file = request.files['fastaFile']
        # Process the file
        # For now, just return the file name as a response
        return jsonify({'file_name': fasta_file.filename})
    return jsonify({'error': 'No file found'})

if __name__ == '__main__':
    app.run(debug=True)