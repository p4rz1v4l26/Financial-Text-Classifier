import pickle
import pandas as pd

model = pickle.load(open(''))
tfidf_vectorizer = pickle.load(open(''))
label_encoder=pickle.load(open(''))

def process(inPath,outPath):
    input_df=pd.read_csv(inPath)
    features=tfidf_vectorizer.transform(input_df['body'])
    
    predictions = model.predict(features)
  # convert output labels to categories
    input_df['category'] = label_encoder.inverse_transform(predictions)
  # save results to csv
    output_df = input_df[['id', 'category']]
    output_df.to_csv(outPath, index=False)

