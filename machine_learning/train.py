from torch.utils.data import DataLoader
from tqdm import tqdm
from lstm_dataset import *
from lstm import *


label_names=['ambulance','apple','bear','bicycle','bird','bus','cat','foot','owl','pig']
dataset = LSTMDataset(data_dir='data1/sketch_datas',label_names=label_names)

train_size=0.8*len(dataset)

train_dataset=[]
val_dataset=[]

for i in range(int(train_size)):
	train_dataset.append(dataset[i])
for i in range(int(train_size),len(dataset)):
	val_dataset.append(dataset[i])

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True,drop_last=True)

model = LSTM(coord_input_dim=2,embed_dim=256,feat_dict_size=100 ,hidden_size=256)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()
train_loss_list = []
train_accuracy_list = []
train_iteration_list = []
 
test_loss_list = []
test_accuracy_list = []
test_iteration_list = []
def train():
    train_correct = 0.0
    train_total = 0.0
    model.train()
    for idx, (coordinate, label, flag_bits, position_encoding) in enumerate(tqdm(train_loader, ascii=True)):
        coordinate = coordinate.cuda()
        label = label.cuda()
        flag_bits = flag_bits.cuda()
        position_encoding = position_encoding.cuda()
            
        flag_bits.squeeze_()
        position_encoding.squeeze_()
        
        optimizer.zero_grad()

        output, _ = model(coordinate, flag_bits, position_encoding)

        batch_loss = loss_function(output, label)
        batch_loss.backward()

        optimizer.step()
        train_predict=torch.max(output.data,1)[1]
        if torch.cuda.is_available():
            train_correct += (train_predict.cuda() == label.cuda()).sum()
        else:
            train_correct += (train_predict == label).sum()
        train_total += label.size(0)
        accuracy = train_correct / train_total * 100.0
        print("Epoch :%d , Batch : %5d , Loss : %.8f,train_correct:%d,train_total:%d,accuracy:%.6f" % (
            epoch + 1, i + 1, batch_loss.item(), train_correct, train_total, accuracy))
def valid():
        model.eval()
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for idx, (coordinate, label, flag_bits, position_encoding) in enumerate(tqdm(val_loader, ascii=True)):
                
                coordinate = coordinate.cuda()
                label = label.cuda()
                flag_bits = flag_bits.cuda()
                position_encoding = position_encoding.cuda()
                flag_bits.squeeze_()
                position_encoding.squeeze_()
                output, _ = model(coordinate, flag_bits, position_encoding)
                batch_loss = loss_function(output, label)
                predicted = torch.max(output.data, 1)[1]
                total += label.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cuda() == label.cuda()).sum()
                else:
                    correct += (predicted == label).sum()
            accuracy = correct / total * 100.0
            test_accuracy_list.append(accuracy)
            test_loss_list.append(batch_loss.item())
            print(" Loss : {}, correct:{}, total:{}, Accuracy : {}".format(batch_loss.item(),
                                                                            correct,
                                                                            total, accuracy))
                


if __name__=='__main__':
      epoch=2
      
      for i in range(epoch):
            train()
            valid()
            torch.save(model,'lstm.pt')