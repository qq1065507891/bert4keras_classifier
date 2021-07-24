from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import tensorflow as tf
import os
import numpy as np


from config import config
from process import build_iteration, read_file, process_text, process_sing_text, process_predict_text
from models import Bert_classifier


def init_model(config):
    model = Bert_classifier(config).build_model()

    AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

    model.compile(
        loss='sparse_categorical_crossentropy',
        # optimizer=Adam(1e-5),  # 用足够小的学习率
        optimizer=AdamLR(learning_rate=1e-3, lr_schedule={
            1000: 1,
            2000: 0.1
        }),
        metrics=['accuracy'],
    )
    return model


def load_model(config):
    model = init_model(config)
    model.load_weights(config['model_file'])
    return model


def train(config):
    train = read_file(config['train_path'])
    test = read_file(config['test_path'])
    dev = read_file(config['dev_path'])

    train, class_weight = process_text(train, config, True)
    test = process_text(test, config)
    dev = process_text(dev, config)

    train_iter = build_iteration(train, config)
    test_iter = build_iteration(test, config)
    dev_iter = build_iteration(dev, config)

    if os.path.exists(config['model_file']):
        model = load_model(config)
    else:
        model = init_model(config)
    cal_backs = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='max'),
        ModelCheckpoint(config['model_file'], monitor='val_loss', verbose=1, save_best_only=True,
                        mode='max', period=1, save_weights_only=True)
    ]

    histroy = model.fit_generator(
        train_iter,
        steps_per_epoch=len(train_iter) // 32,
        epochs=50,
        validation_data=dev_iter,
        validation_steps=len(dev_iter) // 32,
        callbacks=cal_backs,
        class_weight=class_weight,
    )
    model.save_weights(config['model_file2'])
    x = model.evaluate_generator(
        test_iter,
        steps=32)
    print(x)


def predict_single_text(texts, config):
    contents = process_sing_text(texts)
    iter_text = build_iteration(contents, config)
    model = load_model(config)
    pre = model.predict_generator(iter_text, steps=len(iter_text))
    pre = [item.argmax() for item in pre]
    return pre


def predict_text(texts, config):
    contents = process_predict_text(texts)
    iter_text = build_iteration(contents, config)
    model = load_model(config)
    pre = model.predict_generator(iter_text, steps=len(iter_text))
    pre = [item.argmax() for item in pre]
    return pre


if __name__ == '__main__':
    train(config)