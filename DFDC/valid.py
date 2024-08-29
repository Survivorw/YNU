import timm
import torch
import torch.nn as nn
from Classifier import Classifier
from dataset import DeepfakeDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# pretrained_model = timm.create_model("xception", pretrained=True)
pretrained_model = timm.create_model("efficientnetv2_rw_t.ra2_in1k", pretrained=True)
for name, param in pretrained_model.named_parameters():
    if "block11" not in name and "block12" not in name:
        param.requires_grad = False
pretrained_model.fc = nn.Linear(2048, 2)

classifier = Classifier(pretrained_model)
# classifier.load_state_dict(torch.load("xception_block11_block12.pt"))
classifier.load_state_dict(torch.load("efficientnetv2.pt"))

batch_size = 16

transforms_dict = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 使用更小的尺寸
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    ),
}
classifier.eval()

valid_dataset = DeepfakeDataset(r"nerual_texture\val", "val", 8800, transforms_dict)
valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True)

val_loss = 0.0
val_acc = 0.0
loss_func = nn.CrossEntropyLoss()
with torch.no_grad():
    for i, data in enumerate(valid_dataloader):
        images, labels = data

        predicted_labels = classifier(images)  # 确保model是可调用的
        loss = loss_func(predicted_labels, labels)

        val_loss += loss.item()
        val_acc += (
            torch.sum(torch.argmax(predicted_labels, dim=1) == labels).item()
            / batch_size
        )

    val_loss /= len(valid_dataloader)
    val_acc /= len(valid_dataloader)

    print(f"val Loss: {val_loss:.4f} | val Acc.: {val_acc:.4f}")

    # 清空显存
    torch.cuda.empty_cache()
