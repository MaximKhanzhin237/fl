import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Гиперпараметры
batch_size = 32
img_size = 64  # Размер картинок
epochs = 5
lr = 0.001

# Директория с папками цветов
root_dir = './flowers'  # или ваш путь

# Трансформации
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Датасет
dataset = datasets.ImageFolder(root=root_dir, transform=transform)

# Классы (их названия)
flower_names = dataset.classes  # ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
num_classes = len(flower_names)

# Делим на train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Простая CNN под 64x64
class FlowerCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 64->32
        x = self.pool(F.relu(self.conv2(x)))   # 32->16
        x = self.pool(F.relu(self.conv3(x)))   # 16->8
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = FlowerCNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Обучение
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "flower_cnn.pth")
