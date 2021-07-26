import matplotlib.pyplot as plt
import time

from datetime import timedelta
from bert4keras.snippets import sequence_padding


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



def training_curve(loss, acc, val_loss=None, val_acc=None):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(loss, color='r', label='Training Loss')
    if val_loss is not None:
        ax[0].plot(val_loss, color='g', label='Validation Loss')
    ax[0].legend(loc='best', shadow=True)
    ax[0].grid(True)

    ax[1].plot(acc, color='r', label='Training Accuracy')
    if val_loss is not None:
        ax[1].plot(val_acc, color='g', label='Validation Accuracy')
    ax[1].legend(loc='best', shadow=True)
    ax[1].grid(True)


def get_time_idf(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return: 返回使用多长时间
    """
    end_time = time.time()
    time_idf = end_time - start_time
    return timedelta(seconds=int(round(time_idf)))

def build_iteration(dataset, config):
    """生成数据迭代器"""
    iter = DatasetIterator(dataset, config['batch_size'])
    return iter
