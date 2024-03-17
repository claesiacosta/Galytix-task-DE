import warnings
warnings.filterwarnings('ignore')
import logging
import os
from processor import Processor

class Pipeline():
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        
        # Creating an instance of Processor class
        self.processor = Processor()
    
    def run(self):
        # Loading pre-trained word embeddings
        self.processor.load_word2vec_emb()
        
        # Finding closest match for phrases and saving the results
        if not os.path.exists("output\distances.csv"):
            closest_match = self.processor.find_closest_match(save=True)
        
         # Continuously processing user input until 'exit' is entered
        while True:
            user_input = input("Write a phrase ('exit' to quit): ")

            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            # Finding closest match for user input
            closest_match = self.processor.find_closest_match(user_input = user_input)

            if closest_match is not None:
                print("#####################################")
                print("User input: ", closest_match['Phrase'])
                print("Match: ", closest_match['Match'])
                print("Similiarity: ", closest_match['Similiarity'])
                print("#####################################")
            else:
                print("No embeddings found for the input phrase")
                
if __name__ == "__main__":
    try:
        pipeline = Pipeline()
        pipeline.run()
    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
