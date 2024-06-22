from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Charger le pipeline de génération de texte
generator = pipeline('text-generation', model='gpt2')

# Charger le pipeline de traduction (anglais vers français dans ce cas)
translator = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')

def generate_and_translate(input_text):
    # Générer du texte
    generated_outputs = generator(input_text, max_length=100, num_return_sequences=1)
    generated_text = generated_outputs[0]['generated_text']

    # Traduire le texte généré
    translated_outputs = translator(generated_text)
    translated_text = translated_outputs[0]['translation_text']

    return generated_text, translated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data['input_text']
    generated_text, translated_text = generate_and_translate(input_text)
    return jsonify({'generated_text': generated_text, 'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True)
