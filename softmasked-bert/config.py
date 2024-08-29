
class Config(object):
    def __init__(self):
        self.hidden_size = 50
        self.embedding_size = 768
        self.device = 'cuda'
        self.lr = 0.00001
        self.epoch = 5
        self.batch_size =64
