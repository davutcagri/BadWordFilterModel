from flask import Flask, request, jsonify
from model import input_filter, test

app = Flask(__name__)

@app.route('/filter', methods=['POST'])
def bad_word_filter():
    try:
        data = request.json
        messages = data.get('messages', [])

        if not messages:
            return jsonify({'error': 'No messages'}, 400)

        responses = []
        for message in messages:
            response = input_filter(message).astype(str).item()
            responses.append({'message': message, 'response': response})

        return jsonify({'response': responses})

    except Exception as e:
        return jsonify({'error': str(e)})