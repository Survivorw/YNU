from transformers import BertTokenizer, BertModel, BertConfig
from .Bi_Gru import BiGRU
import torch
import torch.nn as nn

config = BertConfig.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert = BertModel.from_pretrained("bert-base-chinese", config=config)

class SoftMaskedBert(nn.Module):
    
    def __init__(self, bert, tokenizer, hidden, layer_n, device) -> None:
        super(SoftMaskedBert,self).__init__()
        self.embedding=bert.embeddings.to(device)
        self.config=bert.config
        embedding_size=self.config.to_dict()['hidden_size']

        self.detector=BiGRU(embedding_size,hidden,layer_n)
        self.corrector=bert.encoder
        mask_token_id=torch.tensor([[tokenizer.mask_token_id]]).to(device)
        self.mask_e=self.embedding(mask_token_id)
        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, input_mask, segment_ids):
            e = self.embedding(input_ids=input_ids, token_type_ids=segment_ids)
            p = self.detector(e)
            e_ = p * self.mask_e + (1-p) * e
            _, _, _, _, \
            _, \
            head_mask, \
            encoder_hidden_states, \
            encoder_extended_attention_mask= self._init_inputs(input_ids, input_mask)
            h = self.corrector(e_,
                            attention_mask=encoder_extended_attention_mask,
                            head_mask=head_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_extended_attention_mask)
            h = h[0] + e
            out = self.softmax(self.linear(h))
            return out, p