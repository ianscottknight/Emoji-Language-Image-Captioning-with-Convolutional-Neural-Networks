import gensim.models as gsm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import numpy as np
import json
import string
import collections
import itertools
import pickle

stop_cache = stopwords.words('english')

e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
w2v = gsm.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

VALID_POS_LIST = ['NN', 'VB', 'JJ']
TOP_N_PREDICTED_EMOJIS = 3 ### BEST
N_EMOJIS_PER_CAPTION = 5
SAVE_FILE_NAME = 'captions_as_emojis_a_{}.json'.format(TOP_N_PREDICTED_EMOJIS)

image_id_to_captions_dic = collections.defaultdict(list)
image_id_to_vectors_dic = collections.defaultdict(list)
image_id_to_emojis_dic = collections.defaultdict(list)

def vectorize(caption, image_id): 
    result = []
    for sent in sent_tokenize(caption):
        for word, pos in pos_tag(word_tokenize(sent)):
            s = word.translate(string.punctuation)
            for valid_pos in VALID_POS_LIST:
                if valid_pos in pos:
                    if s not in stop_cache: 
                        vec = np.zeros_like(w2v['hello'])
                        if 'NN' in pos:
                            try: 
                                vec += w2v[s]
                                result.append(vec) 
                                result.append(vec)
                                result.append(vec) ### We want nouns to be most influential 
                            except: 
                                pass
                        elif 'VB' in pos:
                            try: 
                                vec += w2v[s]
                                result.append(vec) 
                                result.append(vec) ### Then verbs
                            except: 
                                pass
                        else: 
                            try: 
                                vec += w2v[s]
                                result.append(vec) ### Then adjectives
                            except: 
                                pass
                    break
    return result

with open('annotations/captions_val2017.json') as f:
    data = json.load(f)
    for i in data['annotations']: 
        image_id_to_captions_dic[i['image_id']] += [i['caption']]
        image_id_to_vectors_dic[i['image_id']] += vectorize(i['caption'], i['image_id'])
    for i in image_id_to_vectors_dic.keys():
        for vec in image_id_to_vectors_dic[i]:
            image_id_to_emojis_dic[i] += e2v.most_similar(topn=TOP_N_PREDICTED_EMOJIS, positive=[vec])

for image_id in image_id_to_captions_dic.keys(): 
    totals = {}
    for key, value in image_id_to_emojis_dic[image_id]:
        totals[key] = totals.get(key, 0) + value
    result = [(k, totals[k]) for k in sorted(totals, key=totals.get, reverse=True)[:N_EMOJIS_PER_CAPTION]]
    image_id_to_emojis_dic[image_id] = result

final_dict = {}
for i in image_id_to_captions_dic.keys(): 
    final_dict[i] = (image_id_to_captions_dic[i], image_id_to_emojis_dic[i])

h = open(SAVE_FILE_NAME, 'w', encoding='utf-8')
json.dump(final_dict, h, ensure_ascii=False)
h.close()
h = open(SAVE_FILE_NAME, 'w', encoding='utf-8')
json.dump(final_dict, h, ensure_ascii=False)
h.close()
