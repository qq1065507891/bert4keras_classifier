import os
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

from models import BertClassifier
from config import config
from utlis import training_curve, build_iteration, get_time_idf
from process import read_file, process_text, process_single_text, process_predict_text

os.environ['TF_KERAS'] = '1'


def init_model(config):
    model = BertClassifier(config, dropout_rate=0.3).build_model()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy'],
    )
    return model


def load_model(config):
    model = init_model(config)
    model.load_weights(config['model_file'])
    return model


def train(config):
    print('加载数据')
    start_time = time.time()
    train = read_file(config['train_path'])
    test = read_file(config['test_path'])
    dev = read_file(config['dev_path'])

    train, class_weight = process_text(train, config, True)
    test = process_text(test, config)
    dev = process_text(dev, config)

    train_iter = build_iteration(train, config)
    test_iter = build_iteration(test, config)
    dev_iter = build_iteration(dev, config)
    
    end_time = get_time_idf(start_time)
    print('加载完成，用时：', end_time)
    
    if os.path.exists(config['model_file']):
        model = load_model(config)
        print('加载已有模型')
    else:
        model = init_model(config)
        print('初始化模型')
    
    my_callbacks = [
        ModelCheckpoint(config['model_file'], monitor='val_accuracy', mode='max', save_best_only=True,
                        save_weights_only=True,
                        verbose=1),
#         EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)
    ]
    start_time = time.time()
    print('训练开始')
    history = model.fit_generator(
        train_iter,
        steps_per_epoch=len(train_iter) // 32,
        epochs=config['epochs'],
        validation_data=dev_iter,
        validation_steps=len(dev_iter),
#         class_weight=class_weight,
        callbacks=my_callbacks
    )
    
    model.save_weights(config['model_file2'])
    end_time = get_time_idf(start_time)
    print('训练完成, 耗时：', end_time)
    
    training_curve(history.history['loss'], history.history['accuracy'],
                   history.history['val_loss'], history.history['accuracy'])
    
    print(model.evaluate_generator(test_iter, len(test_iter)))


def predict_single_text(text, config):
    contents = process_single_text(text)
    iter_text = build_iteration(contents, config)
    model = load_model(config)
    pre = model.predict_generator(iter_text, steps=len(iter_text)).argmax()
    print(pre)


def predict_text(texts, config):
    start_time = time.time()
    contents = process_predict_text(texts)
    iter_text = build_iteration(contents, config)
    model = load_model(config)
    pre = model.predict_generator(
        iter_text,
        steps=len(iter_text)
    )
    pre = [item.argmax() for item in pre]
    end_time = get_time_idf(start_time)
    print('用时: ', end_time)
    return pre

if __name__=='__main__':
    train(config)
