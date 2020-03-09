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

def format_token_arr(tokenizer,w,r,max_len=75) :
    
    tokenized_texts = []
    tokenized_rects = []
    
    words  = []
    rects  = []
    
    for i in range(len(w)) :
        word   = w[i]
        rect   = r[i]
        
        # Split into tokens by spaces
        # Now split each token into sub-tokens using the tokenizer
        # such that any new sub-tokens receive either a "O" or "I-"
        # label as necessary
        token = tokenizer.tokenize(word)
        if len(token) == 0 :
            continue
        words.extend(token)
        rects.extend([rect for j in range(len(token))])
        
        # Control for sequence length
        if len(words)>max_len :
            tokenized_texts.append(words[0:max_len])
            tokenized_rects.append(rects[0:max_len])
            words = words[max_len:]
            rects = rects[max_len:]
            
    tokenized_texts.append(words)
    tokenized_rects.append(rects)
        
    return tokenized_texts,tokenized_rects

def tag_sentences(tokenizer,model,idx2tag,tts) :
    
    MAX_LEN = 75
    bs      = 32
    device  = 'cpu'

    # Want to factor in batch size here
    # Also need to rearrange rects accordingly
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
#             for_removal = []
#             for i in range(len(res)-1,-1,-1) :
#                 if res[i]['token'].startswith("##") :
#                     res[i-1]['token'] += res[i]['token'][2:]
#                     for_removal.append(res[i])
#             for fr in for_removal :
#                 res.remove(fr)

            ret_json.append(res)

    return ret_json

def filter_partial_tags(tttag,tr) : 

    ttag_out = []
    tr_out   = []
    for j in range(len(tttag)) : 

        ind_remove = []

        words = tttag[j]
        rects = tr[j]

        words_out = []
        rects_out = []

        for i in range(len(words)-1,-1,-1) :
            if words[i]['token'].startswith("##") :
                words[i-1]['token'] += words[i]['token'][2:]
                ind_remove.append(i)

        # Second forward pass to skip over bad tags
        for i in range(len(words)) :
            if i in ind_remove :
                continue
            words_out.append(words[i])
            rects_out.append(rects[i])

        ttag_out.append(words_out)
        tr_out.append(rects_out)
        
    return ttag_out,tr_out

def same_tag_type(ta,tb) : 
    if len(ta) != len(tb) :
        return False
    if ta[2:] == tb[2:] :
        return True
    return False

def combine_tags(tttag,tr) : 
    ttag_out = []
    tr_out   = []
    for j in range(len(tttag)) : 

        ind_remove = []

        words = tttag[j]
        rects = tr[j]
        
        words_out = []
        rects_out = []

        for i in range(len(words)-1,-1,-1) :
            if words[i]['label'].startswith("I-") and same_tag_type(words[i]['label'],words[i-1]['label']) :
                
                print("Matching terms",words[i]['label'],words[i-1]['label'])
                
                words[i-1]['token'] += " "+words[i]['token']
                
                print(words[i-1]['token'])
                
                # print(rects[i-1],rects[i])
                
                # Modify the rect to extend out to the x1,y1 coordinates
                # x0 and y0 should remain the same
                rects[i-1][2] = rects[i][2]
                rects[i-1][3] = rects[i][3]
                
                ind_remove.append(i)

        # Second forward pass to skip over bad tags
        for i in range(len(words)) :
            if i in ind_remove :
                continue
            words_out.append(words[i])
            rects_out.append(rects[i])

        ttag_out.append(words_out)
        tr_out.append(rects_out)
        
    return ttag_out,tr_out    