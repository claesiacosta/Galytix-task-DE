# Data Handling
import gdown
import os
import pandas as pd
import gzip

from gensim.models import KeyedVectors  

from sklearn.metrics.pairwise import cosine_similarity
import logging

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import warnings
warnings.filterwarnings('ignore')

class Processor:
    
    def __init__(self):
        _,_= self.load_data()
    
    def load_data(self):
        # Function to load necessary data for processing
        
        url = "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g"
        output = "data/GoogleNews-vectors-negative300.bin.gz"
        extract_file = "data/GoogleNews-vectors-negative300.bin"
        if not os.path.exists(output):
            # Downloading the word vectors file if not already downloaded
            try:
                gdown.download(url=url, output=output, fuzzy=True, verify = False, quiet=False)
            except Exception as e:
                logging.exception(f"An error occurred: {str(e)}")
        if not os.path.exists(extract_file): 
            try:
                with open(output, 'rb') as inf:
                    compressed_content = inf.read()
                    if len(compressed_content) == 0:
                        raise ValueError("Compressed file is empty.")
                    decompressed_content = gzip.decompress(compressed_content)
                    if len(decompressed_content) == 0:
                        raise ValueError("Decompressed content is empty.")
                    with open(extract_file, 'wb') as tof:
                        tof.write(decompressed_content)
            except Exception as e:
                logging.exception(f"An error occurred: {str(e)}")
        
        # Returning the location of the downloaded file and loaded phrases DataFrame
        return extract_file, pd.read_csv('data/phrases.csv', encoding='latin1')['Phrases']
    
    def remove_stopwords(self, phrase):
        # removing stopwords
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(phrase)
        filtered_phrase = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_phrase)
        
    def remove_duplicates(self, phrases):
        return list(set(phrases))
        
    def load_word2vec_emb(self):
        # Function to load pre-trained word embeddings
        
        logging.info("Loading word2vec embeddings...")
        location,_ = self.load_data()
        
        # Loading word vectors and saving them in a different format for further use
        self.wv = KeyedVectors.load_word2vec_format(location, binary=True, limit=1000000)
        #self.wv.save_word2vec_format('data/vectors.csv', binary=False)

    def cosine_similiarity(self, vec1, vec2):
        # Function to calculate cosine similarity between two vectors
        return cosine_similarity(vec1, vec2)[0][0]
    
    def get_phrase_embedding(self, phrase):
        # Function to obtain embeddings for a given phrase
        emb = []
        for word in phrase.split():
            if word in self.wv:
                emb.append(self.wv[word])
        #return [self.wv[word] for word in phrase.split() if word in self.wv]
        return emb
    
    def find_closest_match(self, user_input = None, save = False):
        # Function to find the closest match for a given input phrase
        distance = []
        logging.info("Loading data...")
        _, phrases = self.load_data()
        
        if user_input is None: 
            user_input = phrases.copy()
        else: 
            user_input = [" ".join(user_input.split())]
            
        logging.info("Processing Phrases...")
        phrases = self.remove_duplicates(phrases)
        
        for user_phrase in user_input:
            for phrase in phrases:
                
                user_phrase_emb = self.get_phrase_embedding(self.remove_stopwords(user_phrase))
                phrase_emb = self.get_phrase_embedding(self.remove_stopwords(phrase))
                
                # Checking if user_phrase_emb is empty
                if not user_phrase_emb:
                    logging.warning(f"Embedding not found for phrase: '{user_phrase}'")
                    return None  # Return None if empty
                
                dist = self.cosine_similiarity(user_phrase_emb, phrase_emb)
                distance.append({'Phrase':user_phrase, 'Match':phrase, 'Similiarity': dist})
                
        df = pd.DataFrame(distance)   
        if save: df.to_csv('output/distances.csv', index=False)
        
        if len(user_input) == 1: # If only one user input, returning the best match
            logging.info("Calculating Similiarity and Best Match...")
            return df.iloc[df['Similiarity'].idxmax()] # Returning the row with maximum similarity
         
        return pd.DataFrame(distance)
                
    