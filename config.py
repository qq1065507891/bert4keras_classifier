config = {
    'train_path': './data/train.txt',
    'test_path': './data/test.txt',
    'dev_path': './data/dev.txt',
    'class_path': './data/class.txt',
    'config_path': './bert_tf/albert_config.json',
    'checkpoint_path': './bert_tf/albert_model.ckpt',
    'dict_path': './bert_tf/vocab.txt',
    'maxlen': 32,
    'batch_size': 32,
    'num_classes': 10
}

class_dict = {
    'finance': 0,
    'realty': 1,
    'stocks': 2,
    'education': 3,
    'science': 4,
    'society': 5,
    'politics': 6,
    'sports': 7,
    'game': 8,
    'entertainment': 9
}