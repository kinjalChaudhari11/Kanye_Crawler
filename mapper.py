# mapper.py
import sys
import io
import re
import nltk
from nltk.stem import PorterStemmer
nltk.download('stopwords', quiet=True)
##doing parts of speech tagging for better accuracy in my sample ?!?!? 
nltk.download('averaged_perceptron_tagger', quiet=True) 
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='latin1')

##Need this for part 5 because it keeeps stemming my words and messing up my sampleeeeeeee ex making kanye into kany :/ 
preserved_words = {'kanye', 'mtv', 'bmi', 'bet', 'donda', 'billboard'}

docid = 1  

for line in input_stream:
    line = line.split(',', 3)[2].strip() 
    words = re.findall(r'\b\w+\b', line.lower())
    word_count = {}

    for word in words:
        word = re.sub(r'[^a-z]', '', word)

        if word and word not in stop_words:

            if word in preserved_words:
                processed_word = word
            else:
                processed_word = stemmer.stem(word)
            
            word_count[processed_word] = word_count.get(processed_word, 0) + 1

    for word, count in word_count.items():
        print(f'{word} {docid} {count}')
    
    docid += 1  



