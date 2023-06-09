import spacy
from sklearn.cluster import KMeans
import pandas as pd

nlp = spacy.load('en_core_web_lg')

def get_dummies(categories, n=5):
    df = pd.DataFrame()
    df['X'] = categories['category'].apply(lambda x: nlp(x).vector)
    X = pd.DataFrame(df['X'].tolist(), index=df['X'].index)
    kmeans = KMeans(n_clusters=n).fit(X)
    df['smart_category'] = kmeans.predict(X).astype('str')
    smart_dummies = pd.get_dummies(df['smart_category'])
    return smart_dummies


if __name__ == '__main__':
    test_input = pd.DataFrame(['cat', 'dog', 'flower', 'tree', 'human', 'child'], columns=['category'])
    print(get_dummies(test_input, 3))
