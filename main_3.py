import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random

# Настройки
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Путь к датасету
root_dir = "./flowers"  # поменяйте на ваш путь, если нужно

# Подсчет классов и создание списка файлов
classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
num_classes = len(classes)

# Трансформации
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Собственный датасет, который можно использовать для разбиения на train/val
class ImagesFromFoldersDataset(Dataset):
    def __init__(self, samples, root, transform=None):
        """
        samples: список кортежей (path, label)
        root: путь к корню датасета (используется для корректной загрузки)
        transform: трансформации (последовательность)
        """
        self.samples = samples
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def gather_samples(root, class_list):
    samples = []
    for label, cls_name in enumerate(class_list):
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                samples.append((os.path.join(cls_dir, fname), label))
    return samples

# Собираем все образцы и делим их на train/val стратифицированно
all_samples = gather_samples(root_dir, classes)

# Стратифицированная разбивка: по классу берем 80% в train, 20% в val
train_indices = []
val_indices = []

random.seed(42)
for c in range(num_classes):
    class_samples = [i for i, (_, lbl) in enumerate(all_samples) if lbl == c]
    random.shuffle(class_samples)
    split = int(0.8 * len(class_samples))
    train_indices.extend(class_samples[:split])
    val_indices.extend(class_samples[split:])

# Построим списки самих путей и меток по выбранным индексам
train_samples = [all_samples[i] for i in train_indices]
val_samples   = [all_samples[i] for i in val_indices]

train_dataset = ImagesFromFoldersDataset(train_samples, root_dir, transform=train_transform)
val_dataset   = ImagesFromFoldersDataset(val_samples, root_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Модель: предобученная ResNet-18
model = models.resnet18(pretrained=True)

# Адаптация последнего слоя под 5 классов
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# Опция: заморозим базовые слои и обучим только "верхний" слой + слой layer4
# Это помогает быстро адаптироваться под новые классы
for name, param in model.named_parameters():
    if "fc" in name or "layer4" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Оптимизатор и критерий
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Обучение
epochs = 25
best_val_acc = 0.0
best_model_path = "flower_cnn.pth"

for epoch in range(epochs):
    # обучение
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    print("epoch start")
    for images, labels in train_loader:
        print("epoch ")
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # валидация
    model.eval()
    val_running_loss = 0.0
    val_total = 0
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs}  "
          f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # сохранить лучшую модель
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Saved best model to {best_model_path}")