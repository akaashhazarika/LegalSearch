from flask import Flask, request, jsonify
from requests import get
import pandas as pd
from bs4 import BeautifulSoup
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
from keras.preprocessing.text import Tokenizer

need_desc = False

app = Flask(__name__)
CORS(app)

def classify(X):
    """
    classification
    :return: 0 or 1
    """

    data = pd.read_csv('LABEL.txt', sep='\t', header=None, encoding="ISO-8859-1")

    model = load_model('CNN_model.h5')
    max_length = 18
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data[1])
    encoded = tokenizer.texts_to_sequences([X])
    print(encoded)
    padded_docs = pad_sequences(encoded, maxlen=max_length, padding='post')
    predictions = model.predict(padded_docs)
    print(padded_docs)
    print(predictions[0][0])

    if predictions[0][0]>0.5:
        return (1,predictions[0][0])
    else:
        return (0,0)

@app.route('/')
def hello_world():
    return 'Hello, World!'


results = [
    {
        "id": "1",
        "url": "",
        "description": ""
    },
    {
        "id": "1",
        "url": "",
        "description": ""
    }
]

@app.route('/search')
def search():
    q = request.args.get('q')
    print(q)
    results = []
    class_res = classify(q)
    if class_res[0] == 1:
        # print("fuck-off this")
        url="https://www.law.cornell.edu/search/site/"
        url=url+q
        response = get(url)
        #print(response.text)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        re=html_soup.find_all('h3')
        print("=====")
        print(len(re))

        for index in range(min(3, len(re))):
            obj = {}

            h3 = re[index]
            a = h3.find_all('a')[0]
            href = a.get('href')

            obj['id'] = index+1
            obj['url'] = href
            text = a.get_text()

            if (need_desc):
                l1 = get(href)
                _html_soup = BeautifulSoup(l1.text, 'html.parser')
                p = _html_soup.find_all("p", {"class": "pro-indent"})
                print(len(p))
                if(len(p) > 0):
                    p_1 = p[0].get_text()
                    desc_1 = p_1[:min(100, len(p_1))] + "..."
                    obj['description'] = desc_1
                    # print(desc_1)
                else:
                    obj['description'] = text
            else:
                obj['description'] = text

            results.append(obj)

        print(results)
        return jsonify({'confidence': str(class_res[1]*100)[:5], 'results': results})

    else:
        print(results)
        return jsonify({'confidence': str(class_res[1]*100)[:5], 'results': []})



if __name__ == '__main__':
    app.run()