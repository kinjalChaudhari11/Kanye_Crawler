import numpy as np
from math import log
import re

class DocumentRelevanceModel:
    def __init__(self, vocab_path):
        self.vocab = []
        self.doc_frequencies = {}
        self.all_doc_ids = set()  # Track all document IDs
        self.load_vocabulary(vocab_path)
        self.IDF = self.compute_IDF()
        self.doc_lengths = self.compute_doc_lengths()
        self.avg_doc_length = np.mean(list(self.doc_lengths.values()))
    
    def load_vocabulary(self, vocab_path):
        """Load vocabulary and document frequencies from reducer output"""
        word_freq = {}
        
        with open(vocab_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue
                    
                word, postings = parts
                postings_list = postings.strip().split()
                
                # Calculate total frequency for the word
                total_freq = 0
                doc_freq = {}
                
                for posting in postings_list:
                    doc_id, count = map(int, posting.strip('()').split(','))
                    total_freq += count
                    doc_freq[doc_id] = count
                    self.all_doc_ids.add(doc_id)  
                
                word_freq[word] = total_freq
                self.doc_frequencies[word] = doc_freq
        
        #  top 200 words by frequency
        self.vocab = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:200]
    
    def compute_doc_lengths(self):
        doc_lengths = {doc_id: 0 for doc_id in self.all_doc_ids}  
        for postings in self.doc_frequencies.values():
            for doc_id, count in postings.items():
                doc_lengths[doc_id] += count
        return doc_lengths
    
    def compute_IDF(self):
        M = len(self.all_doc_ids)
        return [log((M + 1) / (len(self.doc_frequencies.get(word, {})) + 1))
                for word in self.vocab]
    
    def process_query(self, query_text):
        words = query_text.lower().split()
        return {word: words.count(word) for word in set(words)}
    
    def compute_BM25_score(self, query_text, k=3.5, b=0.2):
        query_counts = self.process_query(query_text)
        scores = {doc_id: 0.0 for doc_id in self.all_doc_ids} 
        
        for doc_id in self.all_doc_ids:
            score = 0
            doc_len_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.avg_doc_length)
            
            for word, query_tf in query_counts.items():
                if word in self.vocab:
                    word_idx = self.vocab.index(word)
                    doc_tf = self.doc_frequencies.get(word, {}).get(doc_id, 0)
                    
                    # BM25 term frequency component
                    tf_component = ((k + 1) * doc_tf) / (doc_tf + k * doc_len_norm)
                    score += query_tf * tf_component * self.IDF[word_idx]
            
            scores[doc_id] = score
                
        return scores

def main():
    model = DocumentRelevanceModel('/root/testout/part-00000')
    
    queries = [
        "olympic gold athens",
        "reuters stocks friday",
        "investment market prices"
    ]
    
    my_query = " ".join(model.vocab[:4])
    queries.append(my_query)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        scores = model.compute_BM25_score(query)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Print top 5
        print("Top 5 documents:")
        for doc_id, score in sorted_scores[:5]:
            print(f"Doc {doc_id}: Score {score}")
            
        # Print bottom 5
        print("Bottom 5 documents:")
        for doc_id, score in sorted_scores[-5:]:
            print(f"Doc {doc_id}: Score {score}")

if __name__ == "__main__":
    main()