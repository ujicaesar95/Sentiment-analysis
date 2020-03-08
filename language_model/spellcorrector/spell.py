# based on http://norvig.com/spell-correct.html

import os
import re
import csv
import string
import pickle
import sys
from collections import Counter
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_dir)
import model
from symspellpy.symspellpy import SymSpell, Verbosity  # import the module
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import create_dictionary

class SpellCorrector:

    NEWLINE = '\n'
    SKIP_FILES = {'cmds'}
    CORPUS_PATH  = os.path.join(os.path.dirname(__file__), 'corpus/questions/')

    __control_dict = {}

    def __init__(self, train=False, save=False, corpus_path=CORPUS_PATH, threshold=2):

        self.slang_dict = pickle.load(open(os.path.join(os.path.dirname(__file__), "pickled/_slang_words.p"), "rb"))
        self.slang_dict['dr'] = 'dari'
        self.slang_dict['k'] = 'ke'
        self.slang_dict['sc'] = 'sesar'

        if train:
            create_dictionary.main()
            self.words = self.__words(corpus_path)
            self.counter = self.__counter(self.words)
            self.model = model.LanguageModel(corpus_path=corpus_path)
        else:
            self.words = pickle.load(open(os.path.join(os.path.dirname(__file__), "pickled/_spell_words.p"), "rb"))
            self.counter = pickle.load(open(os.path.join(os.path.dirname(__file__), "pickled/_spell_counter.p"), "rb"))
            self.model = model.LanguageModel(load=True)

        try:
            for key in self.counter:
                if self.counter[key] <= threshold:
                    self.words.remove(key)
        except:
            pass

        self.candidates_dict = {}

        # maximum edit distance per dictionary precalculation
        max_edit_distance_dictionary = 2
        prefix_length = 7

        # create object
        self.sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()
        # load dictionary
        dictionary_path = os.path.join(os.path.dirname(__file__), "corpus/dictionary/dictionary.txt")
        # dictionary_path = os.path.join(os.path.dirname(__file__), "corpus/symspellpy/frequency_dictionary_en_82_765.txt")
        term_index = 0  # column of the term in the dictionary text file
        count_index = 1  # column of the term frequency in the dictionary text file
        if not self.sym_spell.load_dictionary(dictionary_path, term_index, count_index, encoding="utf-8"):
            print("Dictionary file not found")
            return

        if save==True:
            self.save()

    def __read_files(self, path):
        for root, dir_names, file_names in os.walk(path):
            for path in dir_names:
                self.__read_files(os.path.join(root, path))
            for file_name in file_names:
                if file_name not in SpellCorrector.SKIP_FILES:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        lines = []
                        f = open(file_path, encoding='latin-1')
                        for line in f:
                            lines.append(line)
                        f.close()
                        content = SpellCorrector.NEWLINE.join(lines)
                        yield file_path, content

    def __words(self, corpus_path):
        words = []
        for file_name, text in self.__read_files(corpus_path):
            print("process data => "+ file_name)
            words += re.findall(r'\w+', text.lower())
        return words

    def __counter(self, words):
        return Counter(words)

    def __wordProb(self, word):
        "Probability of `word`."
        return self.counter[word] / sum(self.counter.values())

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.__wordProb)

    def candidates(self, word, debug=False):
        "Generate possible spelling corrections for word."
        if self.candidates_dict.get(word):
            return self.candidates_dict[word]
        else:
            # max edit distance per lookup
            # (max_edit_distance_lookup <= max_edit_distance_dictionary)
            max_edit_distance_lookup = 2
            suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
            suggestions = self.sym_spell.lookup(word, suggestion_verbosity, max_edit_distance_lookup)

            # cache it
            if SpellCorrector.__control_dict.get(word) != None:
                candidates_0 = (self.__known([word]) | self.__known(self.__edits1(word)) | self.__known(self.__edits2(word)) | self.__known(self.__edits3(word)) | {SpellCorrector.__control_dict.get(word)} | {word})
            else:
                candidates_0 = (self.__known([word]) | self.__known(self.__edits1(word)) | self.__known(self.__edits2(word)) | self.__known(self.__edits3(word)) | {word})
            candidates_1 = set(suggestion.term for suggestion in suggestions)
            candidates = candidates_0.union(candidates_1)

            # print(candidates)

            self.candidates_dict[word] = candidates
            return candidates

    def __known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.counter)

    def __edits1(self, word):
        "All edits that are one edit away from `word`."
        letters      = 'aiueon'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(inserts)

    def __edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.__edits1(word) for e2 in self.__edits1(e1))

    def __edits3(self, word):
        return (e3 for e1 in self.__edits1(word) for e2 in self.__edits1(e1) for e3 in self.__edits1(e2))

    def save(self, python2=False):
        if python2 is False:
            pickle.dump( self.words, open(os.path.join(os.path.dirname(__file__), "pickled/_spell_words.p"), "wb"))
            pickle.dump( self.counter, open(os.path.join(os.path.dirname(__file__), "pickled/_spell_counter.p"), "wb"))
            self.model.save()
        else:
            pickle.dump( self.words, open(os.path.join(os.path.dirname(__file__), "pickled/_spell_words.p"), "wb"), protocol=2)
            pickle.dump( self.counter, open(os.path.join(os.path.dirname(__file__), "pickled/_spell_counter.p"), "wb"), protocol=2)
            self.model.save()

    # TODO: implement mechanism to calculate lambda for interpolation
    def __trigram_interpolation(self, w1, w2, w3):
        lambda1 = 0.75
        lambda2 = 0.20
        lambda3 = 0.05
        return (lambda1 * self.model.sentence_prob('{} {} {}'.format(w1, w2, w3))) + (lambda2 * self.model.sentence_prob('{} {}'.format(w2, w3))) + (lambda3 * self.model.unigram_prob(w3))

    # TODO: implement mechanism to calculate lambda for interpolation
    def __bigram_interpolation(self, w1, w2):
        lambda1 = 0.80
        lambda2 = 0.20
        return (lambda1 * self.model.sentence_prob('{} {}'.format(w1, w2))) + (lambda2 * self.model.unigram_prob(w2))
    
    def __clean_text(self, words):
        
        cleaned_words = []
        for word in words:
            if word in self.slang_dict:
                word = self.clean_punc(self.slang_dict[word])
            word = re.sub('^days$', 'hari', word)
            word = re.sub('^day$', 'hari', word)
            word = re.sub('^weeks$', 'minggu', word)
            word = re.sub('^week$', 'minggu', word)
            word = re.sub('^months$', 'bulan', word)
            word = re.sub('^month$', 'bulan', word)
            word = re.sub('^years$', 'tahun', word)
            word = re.sub('^year$', 'tahun', word)
            word = re.sub('(?<=\d)tahun', ' tahun ', word).strip()
            word = re.sub('(?<=\d)bulan', ' bulan ', word).strip()
            word = re.sub('(?<=\d)minggu', ' minggu ', word).strip()
            word = re.sub('(?<=\d)hari', ' hari ', word).strip()
            word = re.sub('(?<=\d)jam', ' jam ', word).strip()
            word = re.sub('(?<=\d)detik', ' detik ', word).strip()
            word = re.sub('(?<=\d)th(?=($|\d+))', ' tahun ', word).strip()
            word = re.sub('(?<=\d)thn(?=($|\d+))', ' tahun ', word).strip()
            word = re.sub('(?<=\d)yrs(?=($|\d+))', ' tahun ', word).strip()
            word = re.sub('(?<=\d)bln(?=($|\d+))', ' bulan ', word).strip()
            word = re.sub('(?<=\d)mggu(?=($|\d+))', ' minggu ', word).strip()
            word = re.sub('(?<=\d)mg(?=($|\d+))', ' minggu ', word).strip()
            word = re.sub('(?<=\d)d(?=($|\d+))', ' hari ', word).strip()
            word = re.sub('(?<=\d)w(?=($|\d+))', ' minggu ', word).strip()
            word = re.sub('(?<=\d)wk(?=($|\d+))', ' minggu ', word).strip()
            word = re.sub('(?<=\d)m(?=($|\d+))', ' bulan ', word).strip()
            word = re.sub('(?<=\d)jm(?=($|\d+))', ' jam ', word).strip()
            word = re.sub('(?<=\d)h(?=($|\d+))', ' hari ', word).strip()
            
            # memisahkan keterangan waktu dengan kata sekelilingnya, hariini --> hari ini
            if re.match("(tahun|bulan|minggu|hari|menit|detik)\w+",word) is not None: 
                word = re.search("(tahun|bulan|minggu|hari|menit|detik)(?=\w+)",word).group(0)+' '+re.search("(?:(?<=tahun)|(?<=bulan)|(?<=minggu)|(?<=hari)|(?<=menit)|(?<=detik))\w+",word).group(0)
            
            # mengubah dari ke2 k2 atau ke(angka) k(angka) --> ke 2, k 2
            if re.match("(ke)\d",word) is not None: 
                word = re.search("(ke)(?=\d)",word).group(0)+' '+re.search("(?<=ke)\d",word).group(0)
                
            if re.match("(k)\d",word) is not None: 
                word = re.search("(k)(?=\d)",word).group(0)+' '+re.search("(?<=k)\d",word).group(0)
            
            # mengubah kata dari kata2 --> kata kata
            if re.match("[a-z]+2$",word) is not None: 
                word = word[:-1]+' '+word[:-1]
                
            # mengubah kata dari 2kata --> 2 kata
            if re.match("^\d+[a-z]+$",word) is not None: 
                word = re.search("\d+(?=\w+)",word).group(0)+' '+re.search("(?<=\d)\w+",word).group(0)
            
            # mengubah kata dari kata2nya --> kata katanya
            if re.match("^\w+2\w+$", word) is not None: 
                word = re.search("^\w+(?=2)",word).group(0)+' '+re.search("^\w+(?=2)",word).group(0)+re.search("(?<=2)\w+",word).group(0)
                
            # mengubah kata berakhiran dok kecuali halodok, alodok, sendok, gondok e.g sayadok --> saya dok
            if re.match("(?<!halo)(?<!alo)(?<!sen)(?<!gon)dok$",word) is not None: 
                word = word[:-3]+' '+word[-3:]
                
            # mengubah kata berakhiran dokter kecuali halodokter, alodokter e.g sayadokter --> saya dokter
            if re.match("(?<!halo)(?<!alo)dokter$",word) is not None: 
                word = word[:-6]+' '+word[-6:]
              
            # mengubah kata doksaya --> dok saya
            if re.match("^dok(?!ter)\w+",word) is not None: 
                word = word[:3]+' '+word[3:]
                
            # mengubah kata doktersaya --> dokter saya
            if re.match("^dokter\w+",word) is not None: 
                word = word[:6]+' '+word[6:]
                
            # mengubah kata 20x atau (angka)x --> 20 kali atau (angka) kali
            if re.match("\d+x$",word) is not None: 
                word = word[:-1]+' kali'
            
            if re.match("\w+x$",word) is not None: 
                word = word[:-1]+'nya'
                
            cleaned_words.append(word)
                
        cleaned = ' '.join(cleaned_words).split()
        
        return cleaned

    def normalize(self, sentence):

        cleaned = sentence

        if re.match("[a-zA-Z0-9 ]+\d \d bulan [a-zA-Z0-9 ]+",cleaned) is not None: 
            cleaned = re.search("[a-zA-Z0-9 ]+\d (?=\d bulan [a-zA-Z0-9 ]+)",cleaned).group(0) +\
            re.search("(?<=[a-zA-Z0-9 ]\d \d )bulan [a-zA-Z0-9 ]+",cleaned).group(0)

        if re.match("(?<=\w\s)x(?=\s)", cleaned) is not None:
            cleaned = re.search("[a-zA-Z ]+(?=\sx\s)",cleaned).group(0) + 'nya ' +\
            re.search("(?<=\sx\s)[a-zA-Z ]+",cleaned).group(0)

        cleaned = self.stemmer.stem(cleaned)

        return cleaned
    
    def clean_punc(self, sentence):
        translator = str.maketrans({key: ' ' for key in string.punctuation})
        words = [token.translate(translator).strip() for token in sentence.lower().split()]
        words = ' '.join(words)
        words =  [x.strip().lower() for x in words.split() if x.strip()]
        
        return ' '.join(words)
    
    def generate_candidates(self, sentence):
        # The method translate() returns a copy of the string in which all characters have been translated
        # using table (constructed with the maketrans() function in the str module),
        # optionally deleting all characters found in the string deletechars.
        translator = str.maketrans({key: ' ' for key in string.punctuation})
        words = [token.translate(translator).strip() for token in sentence.lower().split()]
        words = ' '.join(words)
        words =  [x.strip().lower() for x in words.split() if x.strip()] # Hapus seluruh empty char pada list

        valid = {}
        for idx, word in enumerate(words):
            if word not in self.words:
                valid[word.lower()] = 'correction_here'
                
        return valid
        

    def validate(self, sentence, debug=False, return_candidates=False, return_full_words=False):
        # The method translate() returns a copy of the string in which all characters have been translated
        # using table (constructed with the maketrans() function in the str module),
        # optionally deleting all characters found in the string deletechars.
        translator = str.maketrans({key: ' ' for key in string.punctuation})
        words = [token.translate(translator).strip() for token in sentence.lower().split()]
        words = ' '.join(words)
        words =  [x.strip().lower() for x in words.split() if x.strip()] # Hapus seluruh empty char pada list

        full_words = {}
        prediction_candidates = {}
        valid = []
        for word in words:
            if word in self.words:
                valid.append(word.lower())

                full_words[word] = word
            else:
                list_words = self.__clean_text([word])
                valid_ = []
                
                for idx, word_ in enumerate(list_words):
                    candidates = self.candidates(word_.lower())
                    if  idx == 0 :
                        max_word = max([w for w in candidates], key=lambda word_: self.model.unigram_prob(word_))
                        valid_.append(max_word)
                        if debug:
                            print('candidates for '+ word_ +': '+ str(candidates) +', max prob word is '+ max_word.lower())

                    elif idx == 1:
                        max_word = max([w for w in candidates], key=lambda word_: self.__bigram_interpolation(valid_[0], word_))
                        valid_.append(max_word)
                        if debug:
                            print('candidates for '+ word_ +': '+ str(candidates) +', max prob word is '+ max_word.lower())

                    else:
                        max_word = max([w for w in candidates], key=lambda word_: self.__trigram_interpolation(valid_[idx - 2], valid_[idx - 1], word_))
                        valid_.append(max_word)
                        if debug:
                            print('candidates for '+ word_ +': '+ str(candidates) +', max prob word is '+ max_word.lower())
                
                if ' '.join(valid_) == 'terimakasih':
                    valid.append('terima kasih')
                    prediction_candidates[word] = 'terima kasih'
                else:
                    valid.append(' '.join(valid_))
                    prediction_candidates[word] = ' '.join(valid_)
            
                full_words[word] = ' '.join(valid_)
                    
        if return_candidates:
            return prediction_candidates
        if return_full_words:
            return full_words
        else:
            return ' '.join(valid)

    # def validate_texts(use_multithread = True):


