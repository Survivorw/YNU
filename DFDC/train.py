import timm
import torch
import torch.nn as nn
from Classifier import Classifier
from dataset import DeepfakeDataset
from torchvision import transforms
from torch.utils.data import DataLoader


pretrained_model = timm.create_model('xception', pretrained=True)
for name, param in pretrained_model.named_parameters():
    if "block11" not in name and "block12" not in name:
        param.requires_grad = False
pretrained_model.fc = nn.Linear(2048, 2)

classifier = Classifier(pretrained_model)


batch_size = 16  
optim = torch.optim.Adam(classifier.parameters(), lr=0.0002)
loss_func = nn.CrossEntropyLoss()


transforms_dict = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # 使用更小的尺寸
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


train_dataset = DeepfakeDataset(r"nerual_texture\train", "train", 45900, transforms_dict)
test_dataset = DeepfakeDataset(r"nerual_texture\test", "test", 8900, transforms_dict)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)


epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
model = classifier.to(device)  


for epoch in range(1, epochs+1):
    avg_loss = 0.0
    avg_acc = 0.0

    model.train()
    for i, data in enumerate(train_dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        predicted_labels = model(images)  # 确保model是可调用的
        loss = loss_func(predicted_labels, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        avg_loss += loss.item()
        avg_acc += torch.sum(torch.argmax(predicted_labels, dim=1) == labels).item() / batch_size

    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            predicted_labels = model(images)  # 确保model是可调用的
            loss = loss_func(predicted_labels, labels)

            test_loss += loss.item()
            test_acc += torch.sum(torch.argmax(predicted_labels, dim=1) == labels).item() / batch_size

    avg_loss /= len(train_dataloader)
    avg_acc /= len(train_dataloader)
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    print(f"Epoch: {epoch} | Train Loss: {avg_loss:.4f} | Train Acc.: {avg_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc.: {test_acc:.4f}")

    # 清空显存
    torch.cuda.empty_cache()

torch.save(model.state_dict(), "xception_block11_block12.pt")
