from torch.utils.data import Dataset

from PIL import Image
class DeepfakeDataset(Dataset):
    def __init__(self, path: str, train_test_val: str, size:int, transform_dict):
        self.path = path
        self.type = train_test_val
        self.size = size
        self.transform_dict = transform_dict
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # fake is 0 - size/2, real is size/2+1 - size
        if idx > self.size/2:
            idx = idx//2
            img_path = self.path + f"/1/{idx}.jpg"
            image = Image.open(img_path)
            label = 1
        else:

            img_path = self.path + f"/0/{idx}.jpg"
            image = Image.open(img_path)
            label = 0
        
        return self.transform_dict[self.type](image), label