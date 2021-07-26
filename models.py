from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda, Dense, Dropout
import os
os.environ['TF_KERAS'] = '1'

from bert4keras.backend import set_gelu
from bert4keras.models import build_transformer_model


class BertClassifier(object):
    def __init__(self, config, last_activation='softmax', dropout_rate=0):
        self.config = config
        self.last_activation = last_activation
        self.dropout_rate = dropout_rate

    def build_model(self):
        set_gelu('tanh')
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=self.config['config_path'],
            checkpoint_path=self.config['checkpoint_path'],
            model=self.config['model_type'],
            return_keras_model=False,
        )

        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

        # dorpout层
        if 0 < self.dropout_rate < 1:
            output = Dropout(self.dropout_rate)(output)

        output = Dense(
            units=self.config['num_classes'],
            activation=self.last_activation,
            kernel_initializer=bert.initializer
        )(output)

        model = Model(bert.model.input, output)
#         model.summary()
        return model

if __name__ == '__main__':
    from config import config
    model = BertClassifier(config, dropout_rate=0.2).build_model()
    model.summary()
