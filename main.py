import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json

# Определение архитектуры нейросети
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Изменено на размер входа, проверьте эту размерность
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Определение набора данных
class CustomDataset(Dataset):
    def __init__(self, images, annotations, categories, data_dir, transform=None):
        self.images = images
        self.annotations = annotations
        self.categories = categories
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.data_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        label = self.get_label_from_annotation(self.annotations[idx], self.categories)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_label_from_annotation(self, annotation, categories):
        category_id = annotation["category_id"]
        category_name = next(category["name"] for category in categories if category["id"] == category_id)

        if category_name == "drone_h":
            return torch.tensor(0, dtype=torch.long)
        elif category_name == "drone_p":
            return torch.tensor(1, dtype=torch.long)
        else:
            return torch.tensor(-1, dtype=torch.long)

# Параметры
data_dir = "C:/Users/loiiz/PycharmProjects/NN1"
json_file_path = "C:/Users/loiiz/PycharmProjects/NN1/result.json"
test_data_dir = "C:/Users/loiiz/PycharmProjects/NN1/timages"
test_json_file_path = "C:/Users/loiiz/PycharmProjects/NN1/tresult.json"
batch_size = 8
lr = 0.001
num_epochs = 10
num_classes = 2
channels = 3
height = 128
width = 128
inputs = torch.randn((batch_size, channels, height, width))

# Трансформации для обучения (вы можете настроить их в зависимости от ваших требований)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Загрузка данных из JSON
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

# Создание экземпляра нейросети
model = SimpleCNN(num_classes=len(data["categories"]))

# Инициализация оптимизатора
optimizer = optim.Adam(model.parameters(), lr=lr)

# Создание DataLoader для обучающего набора данных
dataset = CustomDataset(data["images"], data["annotations"], data["categories"], data_dir, transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Обучение нейросети
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)

        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Создание DataLoader для тестового набора данных
with open(test_json_file_path, "r") as test_json_file:
    test_data = json.load(test_json_file)

test_dataset = CustomDataset(test_data["images"], test_data["annotations"], test_data["categories"], test_data_dir, transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Тестирование нейросети
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        _, predicted_indices = torch.max(outputs, 1)
        correct += (predicted_indices == labels).sum().item()

        total += labels.size(0)

test_accuracy = correct / total * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')



# Сохранение модели
torch.save(model.state_dict(), 'trained_model.pth')
