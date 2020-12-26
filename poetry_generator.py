from flask import Flask, render_template, request
import torch
from model import model

app = Flask(__name__)
poetry_text = ''
generator_model = model.CharLSTMLoop_hidden()


@app.route('/', methods=['GET', 'POST'])
def index(poetry_text=poetry_text):
    if request.method == 'POST':
        name = request.form.get('name')
        beginning = request.form.get('beginning')
        size = request.form.get('size')
        #if beginning == '':
        #    poetry_text = 'Beginning is required.'
        #    return render_template('style.html', poetry_text=poetry_text)

        if name == 'Shakespeare':
            if match_en(beginning):
                generator_model.load_state_dict(torch.load('/model/model_lstm_hidden.pth'))
                poetry_text = str(model.generate_text_hidden(length=size, initial=beginning))
            else:
                poetry_text = 'The language of the poet is English.'
        if name == 'Пушкин':
            if match_ru(beginning):
                generator_model.load_state_dict(torch.load('/model/model_lstm_hidden.pth'))
                poetry_text = str(model.generate_text_hidden(length=size, initial=beginning))
            else:
                poetry_text = 'The language of the poet is Russian.'

        return render_template('style.html', poetry_text=poetry_text)
    return render_template('style.html', poetry_text=poetry_text)


def match_ru(text, alphabet=set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')):
    return not alphabet.isdisjoint(text.lower())

def match_en(text, alphabet=set('abcdefghijklmnopqrstuvwxyz')):
    return not alphabet.isdisjoint(text.lower())