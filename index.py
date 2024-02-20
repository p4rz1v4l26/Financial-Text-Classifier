import pickle
import pandas as pd

model = pickle.load(open(''))
tfidf_vectorizer = pickle.load(open(''))
label_encoder=pickle.load(open(''))

def process(inPath,outPath):
    input_df=pd.read_csv(inPath)
    features=tfidf_vectorizer.transform(input_df['body'])