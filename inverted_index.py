import pandas as pd

def create_inverted_index(filepath='/root/testout/part-00000'):
    inverted_index = {}
    
    try:
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue
                    
                word, postings = parts
                
                doc_freq_dict = {}
                
                posting_pairs = postings.strip().split()
                for pair in posting_pairs:
                    pair = pair.strip('()')
                    try:
                        doc_id, freq = pair.split(',')
                        doc_freq_dict[int(doc_id)] = int(freq)
                    except ValueError:
                        continue
                
                inverted_index[word] = doc_freq_dict
        
        word_freq = {word: sum(postings.values()) for word, postings in inverted_index.items()}
        sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_200_vocab = [word for word, _ in sorted_vocab[:200]]
        
        return inverted_index, top_200_vocab
                
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return {}, []
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {}, []

if __name__ == '__main__':
    index, vocab = create_inverted_index()
    
    #test: 
    print(f"Total unique terms in index: {len(index)}")
    print(f"Size of vocabulary: {len(vocab)}")
    print("\nSample of vocabulary:", vocab[:10])