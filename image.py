import torch
import torchvision.transforms as transforms
from main import SimpleCNN
from PIL import Image

# Загрузка обученной модели
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Пример тестового изображения (замените на свои данные)
test_image = Image.open('sonydronestill.jpg')

# Преобразование данных
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_data = transform(test_image)
test_data = test_data.unsqueeze(0)  # Добавление размерности пакета (batch dimension)

# Применение модели к тестовым данным
with torch.no_grad():
    output = model(test_data)

# Печать предсказаний
print("Модель предсказывает:", output)

# Разбор вывода модели и получение предсказанного класса
_, predicted_class = torch.max(output, 1)

if predicted_class == 0:
    print("Предсказанный класс: drone_h")

elif predicted_class == 1:
    print("Предсказанный класс: drone_p")
