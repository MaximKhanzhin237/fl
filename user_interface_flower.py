import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

# 1. Определяем класс нейросети (должен совпадать с тем, что вы обучили)
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. Загружаем обученную модель (укажите путь к вашим весам и классам)
@st.cache(allow_output_mutation=True)
def load_model():
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    num_classes = len(class_names)
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load('flower_cnn.pth', map_location='cpu'))
    model.eval()
    return model, class_names

model, class_names = load_model()

# 3. Задаём трансформации, такие же, как при обучении
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 4. Streamlit интерфейс
st.title('Распознавание цветка по изображению 🌸')
st.write('Загрузите фотографию цветка, и нейросеть предположит его вид.')

uploaded_file = st.file_uploader("Выберите изображение цветка...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    
    # Предсказание
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        st.write(f"**Вид цветка:** {class_names[pred.item()]}  ")
        st.write(f"**Уверенность:** {conf.item()*100:.1f}%")
