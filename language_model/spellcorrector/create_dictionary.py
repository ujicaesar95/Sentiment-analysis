import os
from collections import Counter
import re
import json
import pandas as pd

from symspellpy.symspellpy import SymSpell  # import the module

CORPUS_FILE  = os.path.join(os.path.dirname(__file__), 'corpus/questions/corpus.txt')
DICT_FILE  = os.path.join(os.path.dirname(__file__), 'corpus/dictionary/dictionary.txt')
WORD_COUNT_FILE  = os.path.join(os.path.dirname(__file__), 'corpus/dictionary/word_count.csv')

def main():
    # maximum edit distance per dictionary precalculation
    max_edit_distance_dictionary = 2
    prefix_length = 7
    # create object
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    
    # create dictionary using corpus.txt
    if not sym_spell.create_dictionary(CORPUS_FILE):
        print("Corpus file not found")
        return

    f= open(DICT_FILE,"w+")
    for key, count in sym_spell.words.items():
        #print("{} {}".format(key, count))
        f.write("{} {} \r\n".format(key, count))
    f.close()
    print('dictionary file created')

    #create another dictionary file using corpus.txt
    sentence_list = []
    with open(CORPUS_FILE, 'r') as file:
        for line in file.readlines():
            line = re.sub('\n','',line)
            sentence_list.append(line)
    
    corpus = ' '.join(sentence_list)

    word_count = Counter(corpus.split())

    df = pd.DataFrame({'word': list(word_count.keys()), 'count': list(word_count.values())})
    
    df.loc[df['count'].isin(['2','3','4'])].sort_values(by='count').to_csv(WORD_COUNT_FILE,index=False)
    print('word count file created')


if __name__ == "__main__":
    main()