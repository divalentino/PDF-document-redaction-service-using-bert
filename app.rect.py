import flask
from datetime import datetime, timedelta
from flask import Flask, jsonify, abort, request, make_response, url_for
from flask_httpauth import HTTPBasicAuth

from waitress import serve

import pandas as pd
import numpy as np

import torch
# Might need these later for longer strings?
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer
from transformers import BertForTokenClassification

from rect_handling import *

app = Flask(__name__, static_url_path = "")

##########################################################
# Instantiate the BERT model
##########################################################

tag2idx = {'B-art': 0,
 'B-eve': 1,
 'B-geo': 2,
 'B-gpe': 3,
 'B-nat': 4,
 'B-org': 5,
 'B-per': 6,
 'B-tim': 7,
 'I-art': 8,
 'I-eve': 9,
 'I-geo': 10,
 'I-gpe': 11,
 'I-nat': 12,
 'I-org': 13,
 'I-per': 14,
 'I-tim': 15,
 'O': 16}

idx2tag = {}
for key in list(tag2idx.keys()) :
    idx2tag[tag2idx[key]] = key

# Reload the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model     = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))
model.load_state_dict(torch.load("ner.dataset.4.pth",map_location=torch.device('cpu')))
model.eval()
# model.cuda()

def prep_for_tokenization(s) :
    for ip in ".,/;!?'%*-" :
        s = s.replace(ip," "+ip)
    return s

def tokenize_text(s,max_len=75) :
    tokenized_texts = []
    if type(s)==str :
        sentences = [s]
    elif type(s)==list :
        sentences = s
    for sentence in sentences : 
        words  = []
        for word in prep_for_tokenization(sentence).split(" ") : # Any better way to do this?
            # Split into tokens by spaces
            # Now split each token into sub-tokens using the tokenizer
            # such that any new sub-tokens receive either a "O" or "I-"
            # label as necessary
            token = tokenizer.tokenize(word)
            if len(token) == 0 :
                continue
            words.extend(token)
            # Control for sequence length
            if len(words)>max_len :
                tokenized_texts.append(words[0:max_len])
                words = words[max_len:]            
        tokenized_texts.append(words)
    return tokenized_texts
    
@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

# This should probably be a MongoDB or something
# to allow asynchronous reads/writes
queue = {}      

app = flask.Flask("bert_tagger")
app.config["SCHEDULER_API_ENABLED"] = True

#######################################
# Methods to add or remove tasks
#######################################

@app.route('/create_task', methods = ['POST'])
# @auth.login_required
def create_task() :
    
    if not request.json :
        abort(400, 'Did not receive any content in POST request!')

    # Check that we have all requisite fields.
    task = request.json.copy()
    for mfield in ['user','study','path','priority','timestamp'] :
        if mfield not in task :
            abort(400, 'Task creation request missing field: %s'%(mfield))

@app.route('/tag_sentence', methods = ['POST'])
# @auth.login_required
def tag_sentence() :
    
    MAX_LEN = 75
    bs      = 32
    device  = 'cpu'

    if not request.json :
        abort(400, 'Did not receive any content in POST request!')

    # Check that we have all requisite fields.
    sent_json = request.json.copy()

    if 'sentence' not in sent_json.keys() :
        abort(400, 'Did not receive sentence data in POST request!')

    print(sent_json)

    sent   = sent_json['sentence']
    tts    = tokenize_text(sent)

    # Want to factor in batch size here
    ltts = len(tts)
    if ltts>bs :
        nbatch = int(np.ceil(ltts/bs))
        tts    = [tts[(i*bs):((i+1)*bs)] for i in range(nbatch)]
    else : 
        tts = [tts]
        
    assert np.sum([len(itts) for itts in tts])==ltts

    ret_json = [] 
        
    for tokenized_texts in tts : 
    
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        # Used to flag which terms are padding and which are real data
        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

        input_id       = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(attention_masks).to(device)

        outputs           = model(input_id, token_type_ids=None, attention_mask=attention_mask)
        # prediction_scores = outputs[0] # [:2]

        for i in range(len(input_ids)) : 
            prediction_scores = outputs[0][i]
            ovar              = prediction_scores.cpu().detach().numpy()
            pred_labels       = np.array([np.argmax(ovar[j]) for j in range(len(ovar))])
            noid_masks        = (input_ids[i]>0)
            y_pred            = pred_labels[noid_masks]

            try : 
                assert(len(y_pred)==len(tokenized_texts[i]))
            except :
                raise Exception('Tokenized text length differs from predictions!')

            res = [] # {'tokens':[],'labels':[]}
            for j in range(len(y_pred)) :
                res.append({
                    'token' : tokenized_texts[i][j],
                    'label' : idx2tag[y_pred[j]]
                })
                
            # Need to iterate backward and re-stitch fragments
            # starting with ##
            for_removal = []
            for i in range(len(res)-1,-1,-1) :
                if res[i]['token'].startswith("##") :
                    res[i-1]['token'] += res[i]['token'][2:]
                    for_removal.append(res[i])
            for fr in for_removal :
                res.remove(fr)

            ret_json.append(res)

    return jsonify( { 'result': ret_json } )

@app.route('/tag_sentence_rects', methods = ['POST'])
# @auth.login_required
def tag_sentence_rects() :
    MAX_LEN = 75
    bs      = 32
    device  = 'cpu'

    if not request.json :
        abort(400, 'Did not receive any content in POST request!')

    # Check that we have all requisite fields.
    sent_json = request.json.copy()

    if 'tokens' not in sent_json.keys() or 'rects' not in sent_json.keys() :
        abort(400, 'Did not receive sentence data in POST request!')

    print(sent_json)

    words = sent_json['tokens']
    rects = sent_json['rects']

    assert len(words)==len(rects)

    tt,tr = format_token_arr(tokenizer,words,rects)
    tttag = tag_sentences(tokenizer,model,idx2tag,tt)

    for i in range(len(tttag)) :
        assert len(tttag[i])==len(tr[i])

    ttagf,trf = filter_partial_tags(tttag,tr)

    for i in range(len(ttagf)) : 
        print(len(ttagf[i]),len(trf[i]))
        assert len(ttagf[i]) == len(trf[i])

    ttag_comb,tr_comb = combine_tags(ttagf,trf)

    for i in range(len(ttag_comb)) : 
        print(len(ttag_comb[i]),len(tr_comb[i]))
        assert len(ttag_comb[i]) == len(tr_comb[i])

    return jsonify( { 'result': {'tokens' : ttag_comb, 'rects' : tr_comb}} )

if __name__ == '__main__':
    # serve(app, host='0.0.0.0', port=5050)
    app.run(port=5050,debug=True)