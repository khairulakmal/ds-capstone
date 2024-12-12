from keybert import KeyBERT
from transformers import BertModel, AutoTokenizer

def initialize_kw_model():
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    return kw_model

def get_keywords(doc, kw_model):
    if not doc or not isinstance(doc, str):
        raise ValueError("Input document must be a non-empty string.")
    
    keywords = kw_model.extract_keywords(
        doc, 
        keyphrase_ngram_range=(1, 3), 
        stop_words=['is', 'about', 'the', 'a', 'an', 'in', 'for', 'from', 'at', 'of'], 
        top_n=10, 
        use_mmr=True, 
        diversity=0.6)
    return keywords