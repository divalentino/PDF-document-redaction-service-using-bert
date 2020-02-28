import flask
from datetime import datetime, timedelta
from flask import Flask, jsonify, abort, request, make_response, url_for
from flask_httpauth import HTTPBasicAuth

import pandas as pd
import numpy as np

import torch
# Might need these later for longer strings?
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer
from transformers import BertForTokenClassification

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

##########################################################
# Methods when user credentials are required
# (maybe interact with LDAP?)
##########################################################

# auth = HTTPBasicAuth()

# @auth.get_password
# def get_password(username):
#     if username == 'model':
#         return 'model'
#     return None

# @auth.error_handler
# def unauthorized():
#     return make_response(jsonify( { 'error': 'Unauthorized access' } ), 403)
#     # return 403 instead of 401 to prevent browsers from displaying the default auth dialog
    
@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

# This should probably be a MongoDB or something
# to allow asynchronous reads/writes
queue = {}

###############################################

# def sort_queue() :
#     # Sort according to:
#     # Priority: 1 (yes), 0 (no)
#     # Timestamp
#     pass

# def tag_file(id) :
#     task = queue[id]
#     print(task)
#     print("Counting down ...")
#     for i in range(5) :
#         print(i)
#         time.sleep(1)

#     # If successful, remove ID from task queue

# # *** TODO: Add some scheduled tasks for querying for new files and resorting the queue ***
# def process_next_file() :
#     """ Process the next file in the queue """
#     sort_queue()
#     if len(queue)>0 :
#         tag_file(list(queue)[0])

###############################################        

app = flask.Flask("bert_tagger")
app.config["SCHEDULER_API_ENABLED"] = True

# scheduler = APScheduler()
# scheduler.init_app(app)
# scheduler.start()

# scheduler = BackgroundScheduler(daemon=True)
# scheduler.add_job(process_next_file,'interval',seconds=5,max_instances=1)
# # Want to add a job here to periodically refresh the queue with new files
# # in the 
# scheduler.start()

# def test_job():
#     print("test job run")

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

#     # Use hashing to create a unique ID
#     id = hashlib.sha224(str("%s%s%s%s"%(task['user'],task['study'],task['path'],task['timestamp'])).encode()).hexdigest()

#     # Parse the timestamp to allow for use in sorting later
#     ts = dateparser.parse(task['timestamp'])   
#     task['timestamp'] = ts

#     queue[id] = task
#     sort_queue()

#     return jsonify( { 'result': True } ), 201

#     # return jsonify( { 'task': make_public_task(task) } ), 201

# @app.route('/get_queue', methods = ['GET'])
# # @auth.login_required
# def get_queue() :
#     # task = filter(lambda t: t['id'] == task_id, tasks)
#     # if len(task) == 0:
#     #     abort(404)
#     return jsonify( queue )

# @app.route('/delete_task/<int:task_id>', methods = ['DELETE'])
# # @auth.login_required
# def delete_task(task_id) :
#     # task = filter(lambda t: t['id'] == task_id, tasks)
#     # if len(task) == 0:
#     #     abort(404)
#     # tasks.remove(task[0])
#     return jsonify( { 'result': True } )

# @app.route("/test")
# def apscheduler_test() :
#     print("Adding Job")
#     scheduler.add_job(id="101",
#                       func=test_job,
#                       next_run_time=(datetime.now() + timedelta(seconds=10)))
#     return "view", 200

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

    sent            = sent_json['sentence']
    tokenized_texts = tokenize_text(sent)

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Not sure what the best option is for padding here
    # attention masks get ignored during loss calculation anyway - maybe not an issue?
    # tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
    #                     maxlen=MAX_LEN, value=tag2idx["[PAD]"], padding="post",
    #                     dtype="long", truncating="post")

    # Used to flag which terms are padding and which are real data
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

    ret_json = [] 
    for i in range(len(input_ids)) : 

        input_id       = torch.tensor([torch.tensor(input_ids[i]).cpu().numpy()]).to(device)
        attention_mask = torch.tensor([torch.tensor(attention_masks[i]).cpu().numpy()]).to(device)
        
        outputs           = model(input_id, token_type_ids=None, attention_mask=attention_mask)
        prediction_scores = outputs[0] # [:2]

        ovar        = prediction_scores.cpu().detach().numpy()
        pred_labels = np.array([np.argmax(ovar[0][j]) for j in range(len(ovar[0]))])  
        noid_masks  = (input_ids[i]>0) # .detach().cpu().numpy()
        y_pred      = pred_labels[noid_masks]
    
        try : 
            assert(len(y_pred)==len(tokenized_texts[i]))
        except :
            abort(400, 'Tokenized text length differs from predictions!')

        res = [] # {'tokens':[],'labels':[]}
        for j in range(len(y_pred)) :
            res.append({
                'token' : tokenized_texts[i][j],
                'label' : idx2tag[y_pred[j]]
            })

        # Probably need to go back through and re-merge "##" terms here

        ret_json.append(res)

    return jsonify( { 'result': ret_json } )

    # for mfield in ['user','study','path','priority','timestamp'] :
    #     if mfield not in task :
    #         abort(400, 'Task creation request missing field: %s'%(mfield))

if __name__ == '__main__':
    app.run(port=5050)