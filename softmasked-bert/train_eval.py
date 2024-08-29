
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from config import Config
from model.SoftMasked_Bert import SoftMasked_Bert
from pytorch_data import DataSet, collate_fn
config = Config()
device = 'cuda'
train_dataset = DataSet('./data/processed_data/all_same_765376/train.csv')
test_dataset = DataSet('./data/processed_data/all_same_765376/test.csv')
train_generator = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
test_generator = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
model=torch.load("./data/aaa.pt")
model = model.to(device)
optimizer = Adam(model.parameters(), lr=config.lr)
criterion_n, criterion_b = nn.NLLLoss(), nn.BCELoss()
gama = 0.7
for epoch in range(config.epoch):
        avg_loss, total_element = 0, 0
        d_correct, c_correct = 0, 0
        for i, batch_data in enumerate(train_generator):
            batch_input_ids, batch_input_mask, \
            batch_segment_ids , batch_output_ids, batch_labels = batch_data
            batch_input_ids = batch_input_ids.to(device)
            batch_input_mask = batch_input_mask.to(device)
            batch_segment_ids = batch_segment_ids.to(device)
            batch_output_ids = batch_output_ids.to(device)
            batch_labels = batch_labels.to(device)
            output, prob = model(batch_input_ids, batch_input_mask, batch_segment_ids)
            prob = prob.squeeze(2)

            loss_b = criterion_b(prob, batch_labels.float())
            loss_n = criterion_n(output.reshape(-1, output.size()[-1]), batch_output_ids.reshape(-1).long())
            loss = gama * loss_n + (1 - gama) * loss_b
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            output = output.argmax(dim=-1)
            c_correct += sum([output[i].equal(batch_output_ids[i]) for i in range(len(output))])
            prob = torch.round(prob).long()

            d_correct += sum([prob[i].squeeze().equal(batch_labels[i]) for i in range(len(prob))])

            avg_loss += loss.item()
            total_element += len(batch_data)
        

        torch.save(model,"./data/aaa.pt")
        print("EP%d_, avg_loss=" % (epoch), avg_loss / len(train_generator), "d_acc=",
            d_correct / total_element, "c_acc", c_correct / total_element)
        
        total_element = 0
        d_correct, c_correct = 0, 0
        for i, batch_data in enumerate(test_generator):
            batch_input_ids, batch_input_mask, \
            batch_segment_ids , batch_output_ids, batch_labels = batch_data
            batch_input_ids = batch_input_ids.to(device)
            batch_input_mask = batch_input_mask.to(device)
            batch_segment_ids = batch_segment_ids.to(device)
            batch_output_ids = batch_output_ids.to(device)
            batch_labels = batch_labels.to(device)
            output, prob = model(batch_input_ids, batch_input_mask, batch_segment_ids)
            prob = prob.squeeze(2)

            output = output.argmax(dim=-1)

            c_correct += sum([output[i].equal(batch_output_ids[i]) for i in range(len(output))])
            prob = torch.round(prob).long()
            d_correct += sum([prob[i].squeeze().equal(batch_labels[i]) for i in range(len(prob))])
            total_element += len(batch_data)
        model.eval()
        print("d_acc(eval)=", d_correct / total_element, "c_acc(eval)", c_correct / total_element)    
