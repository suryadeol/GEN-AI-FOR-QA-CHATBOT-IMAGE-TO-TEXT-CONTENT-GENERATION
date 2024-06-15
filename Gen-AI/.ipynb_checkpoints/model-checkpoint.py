from flask import Flask, redirect, url_for, request, render_template
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
import os
import re

import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer




app = Flask(__name__)

def clean_text(text):
    # Remove unwanted symbols and characters
    cleaned_text = re.sub(r'\*\*|\n', '', text)
    cleaned_text = cleaned_text.replace('*', '').strip()
    return cleaned_text

def text_info(question):

    
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GOOGLE_API_KEY = 'AIzaSyCKiVblEuDLoo1TxpBgvtslB4MN2hTFD1g'
    genai.configure(api_key=GOOGLE_API_KEY)
 

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    output = clean_text(response.text)

    return output



def text_chat(question):

    
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GOOGLE_API_KEY = 'AIzaSyCKiVblEuDLoo1TxpBgvtslB4MN2hTFD1g'
    genai.configure(api_key=GOOGLE_API_KEY)


    model_chat = genai.GenerativeModel('gemini-pro')
    chat = model_chat.start_chat(history=[])

    response = chat.send_message(question)
    output = clean_text(response.text)

    return output




def bert_ask(question_bert,answer_text):


    model_bert = BertForQuestionAnswering.from_pretrained(r'data/')
    tokenizer = BertTokenizer.from_pretrained(r'data/')

    #tokenize
    encoded_dict = tokenizer.encode_plus(text=question_bert,text_pair=answer_text, add_special=True)
        
        
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = encoded_dict['input_ids']

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # Segment Ids
    segment_ids = encoded_dict['token_type_ids']

    # evaluate
    output = model_bert(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))


    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(output['start_logits'])
    answer_end = torch.argmax(output['end_logits'])

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)


    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')

    return 'Answer: "' + answer + '"'



@app.route('/')
def index():
    return render_template('start.html')


@app.route('/text_gemini')
def text_gemini():
    return render_template('chat.html')



global_input = ""

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    response = text_info(question)

    global global_input

    global_input = response


    return response



@app.route('/bert_chat')
def bert_chat():

    return render_template('bert_chat.html')



@app.route('/bert_process', methods=['POST'])
def bert():

    input_question = request.form['question']

    global global_input

    output_bert = chat_ask(input_question,global_input)


    return output_bert


@app.route('/chat_asst')
def chat_asst():
    return render_template('chat_assitance.html')



@app.route('/chat_process', methods=['POST'])
def chat_assitance():

    input_question = request.form['question']


    output_chat = text_chat(input_question)


    return output_chat







if __name__ == '__main__':
    app.run(debug=True)
