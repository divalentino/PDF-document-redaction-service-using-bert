import sys
import fitz
import requests
import json
import numpy as np

ppd_blue = (115./255., 203./255., 235./255.)

def add_ppd_box(page,rect) : 
    
    page.addRedactAnnot(rect)
    page.apply_redactions()

    # Need to also add "PPD" in the rectangle
    rc    = page.drawRect(rect,color=ppd_blue,fill=ppd_blue)
    shape = page.newShape()

    rc    = shape.insertTextbox(rect, "PPD", fontsize=6)
    shape.commit()

def mark_words_with_rect(page,tags) : 

    tokens = tags['tokens']
    rects  = tags['rects']

    for i in range(len(tokens)) :
        for j in range(len(tokens[i])) : 
            
            value = tokens[i][j]['token']
            label = tokens[i][j]['label']

            for punct in ".,!@?()[]-" :
                value = value.replace(" "+punct,punct)
            
            if label == 'O' :
                continue

            if label[2:].upper() not in ['GEO','TIM','PER'] :
                continue

            # print("Tagging value:",value)
            
            rect  = [float(ri) for ri in rects[i][j]]
            r = fitz.Rect(rect)       # make rect from word bbox
            
            # We've crossed over - try to find the annotation instead
            if rect[0]>rect[2] and rect[3]>rect[1] :
                search_value = value
                print("Looking for multi-line phrase: %s"%(search_value))
                ql = page.searchFor(search_value)
                print(rect)
                print(ql)              
                for iq in range(len(ql)) :                   
                    if abs(float(ql[iq][0])-rect[0])<0.01 :
                        # Hard defaulting to only assuming one line of overlap
                        rect = [ql[iq], ql[iq+1]]
                        print("Got rect:",rect)
                        break

                for ir in rect :
                    add_ppd_box(page,ir)
                        
                continue              
            
            add_ppd_box(page,rect)

fname = "doc2.pdf"

if len(sys.argv)>1 :
    fname = sys.argv[1]

doc   = fitz.open(fname)

for page in doc :
    
    # NOTE: getTextWords also returns rect - we can use this probably
    sentence = " ".join([word[4] for word in page.getTextWords()])
    # print(sentence)
    
    npp    = np.array(page.getTextWords())
    words  = npp[:,4].tolist()
    rects  = npp[:,0:4].tolist()
    
    url   = 'http://127.0.0.1:5050/tag_sentence_rects'
    myobj = {'tokens': words, 'rects' : rects}
    # print(myobj)
    x      = requests.post(url=url,json=myobj)
    tags   = json.loads(x.text)['result']
    
    mark_words_with_rect(page, tags)
    
print("Saving redacted document as: %s"%("redacted-" + doc.name))
doc.save("redacted-" + doc.name)

