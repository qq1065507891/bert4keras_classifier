from sklearn.utils.class_weight import compute_class_weight
import os
os.environ['TF_KERAS'] = '1'
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
import numpy as np

from config import class_dict, config


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        texts, labels = list(), list()
        for line in f.readlines():
            try:
             text, label = line.split('\t')
            except:
                continue
            texts.append(text.strip())
            labels.append(int(label.strip()))
    return [texts, labels]


def process_sing_text(text):
    tokenizer = Tokenizer(config['dict_path'], do_lower_case=True)
    
    token_ids, segment_ids = tokenizer.encode(text, maxlen=config['maxlen'])
    token_ids = sequence_padding([token_ids])
    segment_ids = sequence_padding([segment_ids])
    contents=(token_ids, segment_ids)
    return contents


def process_predict_text(texts):
    tokenizer = Tokenizer(config['dict_path'], do_lower_case=True)
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=config['maxlen'])
        contents.append((token_ids, segment_ids, [0]))
    token_ids = sequence_padding([item[0] for item in contents])
    segment_ids = sequence_padding([item[1] for item in contents])
    content = (token_dis, segment_ids)
    return content


def process_text(datas, config, train=None):
    texts = datas[0]
    labels = datas[1]

    tokenizer = Tokenizer(config['dict_path'], do_lower_case=True)
    contents = []
    for text, label in zip(texts, labels):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=config['maxlen'])
        contents.append((token_ids, segment_ids, [label]))
    if train:
        class_weight = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return contents, class_weight

    return contents
