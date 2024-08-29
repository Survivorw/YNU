import numpy as np
import torch.utils.data as data
import os

class LSTMDataset(data.Dataset):

    def __init__(self, data_dir,label_names):

        self.data_dir=data_dir
        self.data=[]
        for label in label_names:
            self.class_num = label_names.index(label)
            self.coordinate_urls=[url for url in os.listdir(data_dir)]
            for i in range(len(self.coordinate_urls)):
                npy=np.load(os.path.join(self.data_dir,self.coordinate_urls[i])).astype(np.float16)
                for j in range(npy.shape[0]):
                    coordinate = npy[j,:100,:2]
                    if coordinate.shape[0]<100:
                        coordinate=np.pad(coordinate,((0,100-coordinate.shape[0]),(0,0)),'constant',constant_values=(0,0))
                    flag_bits = npy[j,:100,2:4]
                    if flag_bits.shape[0]<100:
                        flag_bits=np.pad(flag_bits,((0,100-flag_bits.shape[0]),(0,0)),'constant',constant_values=(0,0))
                    
                    self.data.append([coordinate,flag_bits,self.class_num])


    def __len__(self):
        length=0
        for i in range(len(self.coordinate_urls)-1):
            npy=np.load(os.path.join(self.data_dir,self.coordinate_urls[i]))
            length+=npy.shape[0]
        return length

    def __getitem__(self, item):
        
        coordinate=np.array(self.data[item][0])
        assert coordinate.shape==(100,2)
        flag_bits=np.array([row[0] for row in self.data[item][1]])
        np.expand_dims(flag_bits,axis=1)
        label=self.data[item][2]

        position_encoding = np.arange(100)
        position_encoding.resize([100, 1])
        
        coordinate = coordinate.astype('float32') 
        
        return (coordinate, label, flag_bits.astype('int'), position_encoding)

