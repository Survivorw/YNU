import torch
import torch.nn as nn

class BiGRU(nn.Module):
    def __init__(self, embedding_size=768, hidden=256, n_layers=10, dropout=0.02):
        super(BiGRU, self).__init__()
        self.BiGRU = nn.GRU(embedding_size, hidden, num_layers=n_layers,
                        bidirectional=True, dropout=dropout, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(hidden*2, 1)

    def forward(self, x):
        out, _ = self.BiGRU(x)
        prob = self.sigmoid(self.linear(out))
        return prob


if __name__ == "__main__":
    model = BiGRU(2, 2, 2)
    text = torch.Tensor([[[1,1],[2,2],[3,3],[4,4]]])
    p = model(text)
    print(p)
    print(text * p)
    print(1-p)