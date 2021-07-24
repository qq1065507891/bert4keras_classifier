from sklearn.utils.class_weight import compute_class_weight
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
    contents = []
    token_ids, segment_ids = tokenizer.encode(text, maxlen=config['maxlen'])
    contents.append((token_ids, segment_ids, [999]))
    return contents


def process_predict_text(texts):
    tokenizer = Tokenizer(config['dict_path'], do_lower_case=True)
    contents = []
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=config['maxlen'])
        contents.append((token_ids, segment_ids, [0]))
    return contents


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


class DatasetIterator(object):
    """
    数据迭代器
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = len(dataset) // batch_size
        self.index = 0

    def _to_tensor(self, datas):
        token_ids = sequence_padding([item[0] for item in datas])
        segment_ids = sequence_padding([item[1] for item in datas])

        y = sequence_padding([item[2] for item in datas])
        return [token_ids, segment_ids], y

    def __next__(self):
        if self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index = 0
            batches = self._to_tensor(batches)
            return batches
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index = self.index + 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches + 1


def build_iteration(dataset, config):
    """生成数据迭代器"""
    iter = DatasetIterator(dataset, config['batch_size'])
    return iter


if __name__ == '__main__':
    data = read_file(config['dev_path'])
    data = process_text(data, config)
    # x = [item[0] for item in data]
    # print(len(x))
    iter = build_iteration(data, config)
    for i in iter:
        print(len(i[0]))
