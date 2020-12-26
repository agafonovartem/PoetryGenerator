from flask import Flask, render_template, request
import torch
import os
from model import model
from model import dataset
from model import generator

path = os.path.split(os.path.realpath(__file__))[0]
app = Flask(__name__)
poetry_text = ''



@app.route('/', methods=['GET', 'POST'])
def index(poetry_text=poetry_text):
    if request.method == 'POST':
        name = request.form.get('name')
        beginning = request.form.get('beginning').lower()
        size = int(request.form.get('size'))
        if beginning == '':
            poetry_text = 'Beginning is required.'
            return render_template('style.html', poetry_text=poetry_text)

        if name == 'Shakespeare':
            if not match_ru(beginning):
                ds = dataset.Dataset(path + '/model/sonnets_upd.txt')
                lstm = model.CharLSTMLoop_hidden(num_tokens=ds.num_tokens)
                if (torch.cuda.is_available()):
                    lstm = lstm.cuda()
                lstm.load_state_dict(torch.load(path + '/model/model_lstm_shakespeare.pth', map_location='cpu'))
                poetry_text = str(generator.generate_text_hidden(length=size, initial=beginning, model=lstm, dataset=ds))
            else:
                poetry_text = 'The language of the poet is English.'
        if name == 'Пушкин':
            if not match_en(beginning):
                ds = dataset.Dataset(path + '/model/sonnets_upd.txt')
                lstm = model.CharLSTMLoop_hidden(num_tokens=ds.num_tokens)
                if (torch.cuda.is_available()):
                    lstm = lstm.cuda()
                lstm.load_state_dict(torch.load(path + '/model/model_lstm_shakespeare.pth', map_location='cpu'))
                poetry_text = str(generator.generate_text_hidden(length=size, initial=beginning, model=lstm, dataset=ds))
            else:
                poetry_text = 'The language of the poet is Russian.'

        return render_template('style.html', poetry_text=poetry_text)
    return render_template('style.html', poetry_text=poetry_text)


def match_ru(text, alphabet=set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')):
    return not alphabet.isdisjoint(text.lower())

def match_en(text, alphabet=set('abcdefghijklmnopqrstuvwxyz')):
    return not alphabet.isdisjoint(text.lower())