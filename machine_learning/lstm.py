
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, coord_input_dim,embed_dim,feat_dict_size,hidden_size) -> None:
        super(LSTM,self).__init__()
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        self.hidden_size = hidden_size
        self.LSTM=nn.LSTM(input_size=coord_input_dim,hidden_size=hidden_size,num_layers=4,dropout=0.5,batch_first=True)
        self.out_layer=nn.Linear(hidden_size*2,10)
    
    def forward(self,coordinate,flag_bits,position_encoding):
        co=self.coord_embed(coordinate)
        fe1=self.feat_embed(flag_bits)
        fe2=self.feat_embed(position_encoding)
        x = co+fe1+fe2

        self.rnn_hidden_feature, _ = self.LSTM(x)
        featur = torch.cat(( self.rnn_hidden_feature[:, -1, : self.hidden_size], 
                            self.rnn_hidden_feature[:, -1, self.hidden_size: ]), 
                            1)
        x = self.out_layer(featur)
        return x, featur

