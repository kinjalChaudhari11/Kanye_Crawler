# reducer.py
import sys

current_word = None
word_data = {}

# Input comes from STDIN (standard input)
for line in sys.stdin:
    line = line.strip().lower()

    try:
        word, doc_id, count = line.split()
        doc_id = int(doc_id)  # Convert doc_id to integer for proper sorting
        count = int(count)
    except ValueError:
        continue

    # If we're still processing the same word
    if current_word == word:
        word_data[doc_id] = word_data.get(doc_id, 0) + count
    else:
        # Output the aggregated data for the previous word
        if current_word:
            # Sort by integer doc_id and create postings list
            postings_list = [f'({doc_id},{count})' for doc_id, count in sorted(word_data.items(), key=lambda x: x[0])]
            print(f'{current_word} {" ".join(postings_list)}')

        # Start processing the new word
        current_word = word
        word_data = {doc_id: count}

# Output the last word's data if there was any
if current_word:
    postings_list = [f'({doc_id},{count})' for doc_id, count in sorted(word_data.items(), key=lambda x: x[0])]
    print(f'{current_word} {" ".join(postings_list)}')