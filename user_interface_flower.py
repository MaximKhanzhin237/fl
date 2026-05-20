import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import os
import importlib
import importlib.util

# 1) Определяем класс нейросети (совпадает с тем, что вы обучали)
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

# 2) Загружаем обученную CNN-модель
@st.cache(allow_output_mutation=True)
def load_model():
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    num_classes = len(class_names)
    model = SimpleCNN(num_classes=num_classes)
    model_path = 'flower_cnn.pth'  # путь к вашим весам CNN
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, class_names

model, class_names = load_model()

# 3) Трансформации
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 4) Функции для загрузки GAN архитектуры и весов

def _load_generator_class():
    """
    Поиск класса Generator в известных местах.
    Не пытаемся загрузить .pth как модуль.
    Возможные варианты:
    - модуль flower_gan_generator.py (класс Generator)
    - модули: generator_model, gan_models, models.generator и др.
    """
    candidates = [
        "flower_gan_generator",
        "generator_model",
        "gan_models",
        "models.generator",
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "Generator"):
                return getattr(mod, "Generator")
        except Exception:
            pass

    # Попытка загрузить локальный файл flower_gan_generator.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gen_py = os.path.join(base_dir, "flower_gan_generator.pth")
    if os.path.exists(gen_py):
        spec = importlib.util.spec_from_file_location("flower_gan_generator_local", gen_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "Generator"):
            return getattr(mod, "Generator")

    return None

def _init_generator_from_weights(weights_path: str, device: torch.device, num_classes: int):
    GeneratorClass = _load_generator_class()
    if GeneratorClass is None:
        return None, None  # архитектура не найдена

    z_dim = getattr(GeneratorClass, "z_dim", getattr(GeneratorClass, "Z_DIM", 100))
    G = GeneratorClass(z_dim, num_classes).to(device)
    G.eval()
    try:
        G.load_state_dict(torch.load(weights_path, map_location=device))
    except Exception as e:
        st.error(f"Не удалось загрузить веса Generator GAN: {e}")
        return None, None

    return G, z_dim

def _load_discriminator_class():
    """
    Попытка найти архитектуру Discriminator аналогично Generator.
    """
    candidates = [
        "flower_gan_discriminator",
        "discriminator_model",
        "gan_discriminator",
        "models.discriminator",
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "Discriminator"):
                return getattr(mod, "Discriminator")
        except Exception:
            pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    disc_py = os.path.join(base_dir, "flower_gan_discriminator.pth")
    if os.path.exists(disc_py):
        spec = importlib.util.spec_from_file_location("flower_gan_discriminator_local", disc_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "Discriminator"):
            return getattr(mod, "Discriminator")

    return None

def _init_discriminator_from_weights(weights_path: str, device: torch.device):
    DiscClass = _load_discriminator_class()
    if DiscClass is None:
        return None
    # Попытка инстанциировать дискриминатор. Поддерживаем несколько стратегий.
    try:
        D = DiscClass().to(device)
        D.eval()
        D.load_state_dict(torch.load(weights_path, map_location=device))
        return D
    except Exception as e:
        # Попробуем простую попытку без аргументов и с передачей обратно, если сигнатура другая
        try:
            import inspect
            sig = inspect.signature(DiscClass)
            # Если конструктор требует аргументов, попробуем заполнить значениями по умолчанию
            params = []
            for pname, p in sig.parameters.items():
                if p.default is not p.empty:
                    params.append(p.default)
                else:
                    # Невозможно автоматически заполнить
                    raise ValueError("Cannot auto-instantiate discriminator without defaults")
            D = DiscClass(*params).to(device)
            D.eval()
            D.load_state_dict(torch.load(weights_path, map_location=device))
            return D
        except Exception as e2:
            st.error(f"Не удалось загрузить дискриминатор: {e2}")
            return None

# 5) Пути к весам GAN (локально рядом с кодом)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GAN_WEIGHTS = os.path.join(BASE_DIR, "flower_gan_generator.pth")
DISCR_WEIGHTS = os.path.join(BASE_DIR, "flower_gan_discriminator.pth")

gan_weights_path = DEFAULT_GAN_WEIGHTS  # путь до весов генератора (почему-то упростим)

# 6) Инициализация GAN по локальным файлам (авто-загрузка)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_G = None
_z_dim = None
_G, _z_dim = _init_generator_from_weights(gan_weights_path, _device, num_classes=len(class_names))

# Попытка загрузить дискриминатор (если есть) и веса
_Discriminator = None
_Discriminator = _init_discriminator_from_weights(DISCR_WEIGHTS, _device)

# 7) UI: загружаем изображение цвета (уникальный ключ)
st.title('Распознавание цветка и генерация 3 картинок GAN')
st.write('Загрузите фотографию цветка, модель распознает вид, а GAN сгенерирует 3 картинки цветов этого же вида.')

flower_image_uploaded = st.file_uploader(
    "Выберите изображение цветка...",
    type=['jpg', 'jpeg', 'png'],
    key="flower_image_upload"  # уникальный ключ
)

if flower_image_uploaded is not None:
    image = Image.open(flower_image_uploaded).convert('RGB')
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    # Предсказание
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        st.write(f"**Вид цветка:** {class_names[pred.item()]}  ")
        st.write(f"**Уверенность:** {conf.item()*100:.1f}%")

    # 8) Генерация 3 картинок GAN (если архитектура найдена)
    if _G is not None:
        class_idx = int(pred.item())
        class_name = class_names[class_idx]

        try:
            generated_images = []
            with torch.no_grad():
                for _ in range(3):
                    z = torch.randn(1, _z_dim, device=_device)
                    label_tensor = torch.tensor([class_idx], dtype=torch.long, device=_device)
                    fake = _G(z, label_tensor)  # [1, C, H, W]
                    img_tensor = (fake.squeeze(0) + 1.0) / 2.0
                    img_tensor = img_tensor.clamp(0, 1)
                    pil_img = transforms.ToPILImage()(img_tensor.cpu())
                    generated_images.append(pil_img)

            captions = [f"{class_name} #{i+1}" for i in range(3)]
            cols = st.columns(len(generated_images))
            for idx, pil_img in enumerate(generated_images):
                cols[idx].image(pil_img, caption=captions[idx], use_column_width=True)
        except Exception as e:
            st.error(f"Не удалось сгенерировать изображения GAN: {e}")
    else:
        st.warning("Не найдена архитектура Generator. Убедитесь, что файл flower_gan_generator.py присутствует в проекте или доступен локально.")
else:
    st.info("Пожалуйста, загрузите изображение цветка выше.")
