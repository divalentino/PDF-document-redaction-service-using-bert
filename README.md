# flask-testing-BertForTokenClassification-transformers

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

```
