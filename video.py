import torch
import torchvision.transforms as transforms
from main import SimpleCNN
from PIL import Image
import cv2
import time

# Загрузка модели
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Пример тестового видеоролика (замените на свои данные)
video_path = '1_1.mp4'
cap = cv2.VideoCapture(video_path)

# Преобразование данных
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Словарь для соответствия числовых значений классов и их строковых представлений
class_mapping = {0: "drone_h", 1: "drone_p"}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в изображение
    test_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    test_data = transform(test_image)
    test_data = test_data.unsqueeze(0)  # Добавление размерности пакета (batch dimension)

    # Применение модели к тестовым данным
    with torch.no_grad():
        output = model(test_data)

    # Разбор вывода модели и получение предсказанного класса
    _, predicted_class = torch.max(output, 1)

    # Получение строкового представления класса
    predicted_class_str = class_mapping[predicted_class.item()]

    # Вывод таймкода и предсказанного класса
    time_in_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    print(f"Таймкод: {time_in_seconds:.2f} сек, Предсказанный класс: {predicted_class_str}")

cap.release()
cv2.destroyAllWindows()
