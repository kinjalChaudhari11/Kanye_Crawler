import numpy as np
from math import log

class ProbRetrieval:
    def __init__(self, vocab_path, lambda_param=0.5):
        self.vocab = []
        self.doc_frequencies = {}
        self.all_doc_ids = set()
        self.lambda_param = lambda_param
        self.collection_probabilities = {}  # p(w|C)
        self.total_collection_words = 0
        self.doc_lengths = {}
        self.load_vocabulary(vocab_path)
        
    def load_vocabulary(self, vocab_path):
        word_freq = {}
        total_word_counts = {} 
        
        with open(vocab_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue
                    
                word, postings = parts
                postings_list = postings.strip().split()
                
                total_freq = 0
                doc_freq = {}
                
         
                for posting in postings_list:
                    doc_id, count = map(int, posting.strip('()').split(','))
                    total_freq += count
                    doc_freq[doc_id] = count
                    self.all_doc_ids.add(doc_id)
                    
                    if doc_id not in self.doc_lengths:
                        self.doc_lengths[doc_id] = 0
                    self.doc_lengths[doc_id] += count
                
                word_freq[word] = total_freq
                self.doc_frequencies[word] = doc_freq
                total_word_counts[word] = total_freq
                self.total_collection_words += total_freq
        
    
        self.vocab = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:200]
        for word in self.vocab:
            self.collection_probabilities[word] = total_word_counts[word] / self.total_collection_words
    
    def process_query(self, query_text):
        return {word: 1 for word in query_text.lower().split()}  
    
    def compute_document_score(self, query_text):
        query_terms = self.process_query(query_text)
        scores = {doc_id: 0.0 for doc_id in self.all_doc_ids}
        
        for doc_id in self.all_doc_ids:
            doc_length = self.doc_lengths[doc_id]
            score = 0.0
            
            for term in query_terms:
                if term in self.vocab:
                    term_freq = self.doc_frequencies.get(term, {}).get(doc_id, 0)

                    collection_prob = self.collection_probabilities.get(term, 1/self.total_collection_words)
                    
                    #  Jelinek-Mercer 
                    prob = (1 - self.lambda_param) * (term_freq / doc_length) + \
                           self.lambda_param * collection_prob
                    
                    # adding log probability to score
                    if prob > 0:  # Avoid log(0)
                        score += log(prob)
                
            scores[doc_id] = score
        
        return scores

def main():
    # Initialize model with lambda=0.5 
    model = ProbRetrieval('/root/testout/part-00000', lambda_param=0.5)
    
    queries = [
        "olympic gold athens",
        "reuters stocks friday",
        "investment market prices"
    ]
    
    my_query = " ".join(model.vocab[:4])  
    queries.append(my_query)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        scores = model.compute_document_score(query)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # top 5
        print("Top 5 documents:")
        for doc_id, score in sorted_scores[:5]:
            print(f"Doc {doc_id}: Score {score}")
            
        #bottom 5
        print("Bottom 5 documents:")
        for doc_id, score in sorted_scores[-5:]:
            print(f"Doc {doc_id}: Score {score}")

if __name__ == "__main__":
    main()