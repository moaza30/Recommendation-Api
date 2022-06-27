from flask import Flask, jsonify, request
from scipy import sparse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

book = pd.read_csv('books.csv')
tfidf_books = sparse.load_npz("tfidf_books.npz")

cs_books = cosine_similarity(tfidf_books, tfidf_books)
indices = pd.Series(book['Title'])


def recommend(title, cosine_sim = cs_books):
    recommended_books = []
    similarities = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    new_score = list(enumerate(score_series))
    new_score = new_score[1:6]
    top_5_indices = list(score_series.iloc[1:6].index)
    for i in top_5_indices:
        recommended_books.append(list(book['Title'])[i])
    return recommended_books



@app.route('/api/recommend/<name>', methods=['GET', 'POST'])
def bookRecommend(name):
    if request.method == 'GET':
        books = recommend(str(name))

    return {"books": books}, 200


if __name__ == '__main__':
    app.run(debug=True)

