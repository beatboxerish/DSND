import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Layout, Figure
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    tokenizes a sentence after normalizing it and returns lemmatized 	 tokens.
    '''
    # normalizing, tokenizing, lemmatizing the sentence
    sentence = re.sub('\W',' ', text)
    tokens = word_tokenize(sentence.lower().strip())
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(i,'n') for i in tokens]
    tokens = [lemmatizer.lemmatize(i,'v') for i in tokens]
    return tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages_clean', engine)

# load model
model = joblib.load("models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    genre_composition = df.loc[:,'genre':].groupby('genre').sum()
    direct_composition = genre_composition.iloc[0].sort_values()
    news_composition = genre_composition.iloc[1].sort_values()
    social_composition = genre_composition.iloc[2].sort_values()
    # create visuals
    trace1 = [Bar(x=genre_names,
                  y=genre_counts)]

    layout1 = Layout(title= 'Distribution of Message Genres',
                     yaxis= {'title': "Count"},
                     xaxis= {'title': "Genre"})
    fig1 = Figure(data = trace1, layout = layout1)
    
    trace2 = [Bar(y=direct_composition,
                  x=direct_composition.index.tolist())]

    layout2 = Layout(title= 'Labels of Messages under Direct Genre',
                     yaxis= {'title': "Count"}
                     )
    fig2 = Figure(data = trace2, layout = layout2)

    trace3 = [Bar(y=news_composition,
                  x=news_composition.index.tolist())]

    layout3 = Layout(title= 'Labels of Messages under News Genre',
                     yaxis= {'title': "Count"}
                    )
    fig3 = Figure(data = trace3, layout = layout3)

    trace4 = [Bar(y=social_composition,
                  x=social_composition.index.tolist())]

    layout4 = Layout(title= 'Labels of Messages under Social Genre',
                     yaxis= {'title': "Count"},
                     )
    fig4 = Figure(data = trace4, layout = layout4)

    graphs = [fig1, fig2, fig3, fig4]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
