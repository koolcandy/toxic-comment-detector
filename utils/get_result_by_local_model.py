from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import joblib
from cleantext import clean

# encoding:utf-8


from utils import config

def clean_text(text):
    """
    Cleans the given text by removing unwanted elements such as unicode characters, URLs, emails, phone numbers, numbers, digits, currency symbols, and punctuation.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    return clean(text,
        fix_unicode=True,               
        to_ascii=True,                  
        lower=True,                     
        no_line_breaks=False,           
        no_urls=False,                  
        no_emails=False,                
        no_phone_numbers=False,         
        no_numbers=False,               
        no_digits=False,                
        no_currency_symbols=False,      
        no_punct=False,                 
        replace_with_punct="",          
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       
    )

def main(comment):
    """
    Get the toxicity prediction result for a given comment using local models.

    Parameters:
    comment (str): The comment for which toxicity prediction is required.

    Returns:
    dict: A dictionary containing the toxicity prediction probabilities for different labels.
          The keys of the dictionary are the labels ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'),
          and the values are the corresponding prediction probabilities.
    """
    tfd = joblib.load(os.path.join(config.work_dir, 'model', 'tfidf_vectorizer.pkl'))

    cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    cleaned_comment = clean_text(comment)
    comment_data = tfd.transform([cleaned_comment])

    predictions = {}
    for label in cols:
        lr = joblib.load(os.path.join(config.work_dir, 'model', f'{label}_model.pkl'))
        predictions[label] = lr.predict_proba(comment_data)[:,1].tolist()[0]
    print('Local model result:', predictions)
    return predictions