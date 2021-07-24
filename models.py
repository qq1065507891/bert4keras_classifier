from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from keras.layers import Lambda, Dense, Dropout


class Bert_classifier():
    def __init__(self, config, last_activation='softmax', dropout_rate=0):
        self.config = config
        self.last_activation = last_activation
        self.dropout_rate = dropout_rate

    def build_model(self):
        set_gelu('tanh')  # 切换gelu版本
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=self.config['config_path'],
            checkpoint_path=self.config['checkpoint_path'],
            model=self.config['model_type'],
            return_keras_model=False,
        )

        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

        if 0 < self.dropout_rate < 1:
            output = Dropout(self.dropout_rate)(output)
        output = Dense(
            units=self.config['num_classes'],
            activation='softmax',
            kernel_initializer=bert.initializer
        )(output)

        model = keras.models.Model(bert.model.input, output)
        # model.summary()
        return model
