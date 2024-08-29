import torch
from config import Config
from model.SoftMasked_Bert import SoftMasked_Bert
from model.BiGRU_Detector import BiGRU_Detector
from transformers import BertTokenizer
import numpy as np

config=Config()
device=config.device
def Generator(Original:str):
    tokenizer=BertTokenizer.from_pretrained("bert-base-chinese")
    model=torch.load('aaa copy.pt').to(device)

    Original=list(Original.strip())
    input_token=['[CLS]'] + Original + ['[SEP]']
    input_id=tokenizer.convert_tokens_to_ids(input_token)
    input_mask = [1] * len(input_id)
    segment_ids = [0] * len(input_id)


    input_id = torch.from_numpy(np.asarray(input_id)).long().unsqueeze(0).to(device)
    input_mask = torch.from_numpy(np.asarray(input_mask)).long().unsqueeze(0).to(device)
    segment_ids = torch.from_numpy(np.asarray(segment_ids)).long().unsqueeze(0).to(device)

    output,_=model(input_id,input_mask,segment_ids)

    output=output.argmax(dim=-1)

    text=tokenizer.decode(output.squeeze(0)).replace(" ", "")[5:-5]
   
    return text

