# GAN для условной генерации цветов (cGAN)
# Требуется PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Путь к данным вашего датасета, аналогично вашему CNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Ваша существующая модель CNN для распознавания (можно подключить позже)
# from ваш_модуль import FlowerCNN  # если вы вынесете CNN отдельно

# Устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры GAN
z_dim = 100
num_classes = 5  # количество видов цветов в вашем датасете
lr_g = 0.0002
lr_d = 0.0002
betas = (0.5, 0.999)
batch_size = 4
epochs_gan = 50  # можно начать с 25-50

# Трансформации и датасет (те же, что вы используете для CNN)
img_size = 64
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

root_dir = './flowers'
dataset = datasets.ImageFolder(root=root_dir, transform=transform)
train_size = int(0.8 * len(dataset))
train_dataset, _ = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Генератор
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, z_dim)
        self.fc = nn.Linear(z_dim + z_dim, 256 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 8x8 -> 16x16
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 16x16 -> 32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)     # 32x32 -> 64x64

    def forward(self, z, labels):
        # z: [B, z_dim], labels: [B]
        label_vec = self.label_emb(labels)  # [B, z_dim]
        x = torch.cat([z, label_vec], dim=1)  # [B, 2*z_dim]
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)  # [B,256,8,8]
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))  # [-1,1]
        return x

# Дискриминатор
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)   # 64 -> 32x32
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # 16x16 -> 8x8

        self.label_emb = nn.Embedding(num_classes, 256)  # для добавления условной информации

        self.fc = nn.Linear(256 * 8 * 8, 1)

    def forward(self, img, labels):
        x = F.leaky_relu(self.conv1(img), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)  # [B,256,8,8]

        # Вектор-метка в такой же размерности, чтобы можно "слить" с признаками
        label_vec = self.label_emb(labels)          # [B,256]
        label_map = label_vec.view(-1, 256, 1, 1).expand(-1, 256, 8, 8)  # [B,256,8,8]

        x = x + label_map  # условное слияние
        x = x.view(-1, 256 * 8 * 8)
        out = self.fc(x)
        return out  # логиты для BCEWithLogitsLoss

# Инициализация моделей
G = Generator(z_dim, num_classes).to(device)
D = Discriminator(num_classes).to(device)
print("Инициализация моделей прошла")
# Оптимизаторы
opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=betas)
opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=betas)

# Функция потерь
criterion = nn.BCEWithLogitsLoss()

# Вспомогательная функция: денормализация для визуализации
def denorm(t):
    return (t + 1) / 2  # из [-1,1] в [0,1]
print("начало обучения")
print(train_loader)
# Обучение GAN
for epoch in range(epochs_gan):
    print("epoch start")
    g_loss_total = 0.0
    d_loss_total = 0.0
    for real_imgs, real_labels in train_loader:
        print("real_imgs")
        real_imgs = real_imgs.to(device)
        real_labels = real_labels.to(device)
        batch_size = real_imgs.size(0)

        # Обучение дискриминатора на реальных изображениях
        D.zero_grad()
        label_real = torch.ones(batch_size, 1, device=device)
        output_real = D(real_imgs, real_labels)
        loss_real = criterion(output_real, label_real)

        # Обучение дискриминатора на поддельных (генерируем с произвольной меткой)
        z = torch.randn(batch_size, z_dim, device=device)
        fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        fake_imgs = G(z, fake_labels).detach()
        label_fake = torch.zeros(batch_size, 1, device=device)
        output_fake = D(fake_imgs, fake_labels)
        loss_fake = criterion(output_fake, label_fake)

        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        opt_D.step()

        # Обучение генератора
        G.zero_grad()
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = G(z, fake_labels)
        output = D(fake_imgs, fake_labels)
        loss_G = criterion(output, label_real)  # стремимся, чтобы дискриминатор думал, что подделка реальна
        loss_G.backward()
        opt_G.step()

        g_loss_total += loss_G.item()
        d_loss_total += loss_D.item()

    print(f"Epoch [{epoch+1}/{epochs_gan}]  D_loss: {d_loss_total/len(train_loader):.4f}  G_loss: {g_loss_total/len(train_loader):.4f}")

# Сохранение обученных весов
torch.save(G.state_dict(), "flower_gan_generator.pth")
torch.save(D.state_dict(), "flower_gan_discriminator.pth")


