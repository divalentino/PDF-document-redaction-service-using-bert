# flask-testing-BertForTokenClassification-transformers

A simple Flask-based web app implementing huggingface's `BertForTokenClassification`, which receives a request containing a sentence (or string of sentences), and returns arrays of tokens with their associated BIO-formatted labels. Based off the approach taken here: https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/

## Usage

```
python app.py
```

Sentences can be sent e.g. from `curl` as follows,

```
sentence="Warner told supporters outside of the University of Virginia Friday that he will not seek a sixth term in the 2008 elections ."
curl localhost:5050/tag_sentence -d '{"sentence": "'"${sentence}"'"}' -H 'Content-Type: application/json'
```

which will return a JSON of tokens with their associated labels, e.g.

```
{"result":[[{"label":"B-per","token":"warner"},{"label":"O","token":"told"},{"label":"O","token":"supporters"},{"label":"O","token":"outside"},{"label":"O","token":"of"},{"label":"O","token":"the"},{"label":"B-org","token":"university"},{"label":"I-org","token":"of"},{"label":"I-org","token":"virginia"},{"label":"B-tim","token":"friday"},{"label":"O","token":"that"},{"label":"O","token":"he"},{"label":"O","token":"will"},{"label":"O","token":"not"},{"label":"O","token":"seek"},{"label":"O","token":"a"},{"label":"O","token":"sixth"},{"label":"O","token":"term"},{"label":"O","token":"in"},{"label":"O","token":"the"},{"label":"B-tim","token":"2008"},{"label":"O","token":"elections"},{"label":"O","token":"."}]]}
```

## Training your own model

See `notebooks/bert_testing_transformers.ipynb` for a Jupyter notebook which can be used to reproduce the model file, e.g. on Google Colab. Note that the file `ner_dataset.csv` can be found easily on Kaggle.