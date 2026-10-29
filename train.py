import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

print("1. Загрузка данных...")
# Читаем наш датасет
df = pd.read_csv("yield_df_cleaned.csv")

# Удаляем лишнюю колонку с индексами, если она есть
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print("2. Подготовка данных (Препроцессинг)...")
# Нейросети не понимают текст (названия стран и культур). 
# Нам нужно превратить их в числа (Label Encoding)
le_area = LabelEncoder()
le_item = LabelEncoder()

df['Area_encoded'] = le_area.fit_transform(df['Area'])
df['Item_encoded'] = le_item.fit_transform(df['Item'])

# Выбираем признаки (X), на которых будем учиться
# Год, Осадки (мм), Пестициды (тонны), Температура, Страна(код), Культура(код)
X = df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area_encoded', 'Item_encoded']].values

# Выбираем то, что нужно предсказать (Y) - Урожайность
# В этом датасете она в гектограммах на гектар (hg/ha), мы переведем в тонны на гектар (t/ha) для удобства
y = (df['hg/ha_yield'] / 10000).values 

# Масштабируем признаки (чтобы большие числа, вроде 2000 мм осадков, не "оглушили" нейросеть)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разбиваем данные на обучающую (80%) и тестовую (20%) выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Переводим в тензоры PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

print("3. Создание нейросети...")
# Создаем архитектуру нейросети (полносвязная сеть - MLP)
class YieldNet(nn.Module):
    def __init__(self, input_size):
        super(YieldNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)

# Инициализируем модель (у нас 6 признаков на входе)
model = YieldNet(input_size=6)

# Настраиваем обучение: функция ошибки (MSE) и оптимизатор (Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("4. Старт обучения (Эпохи)...")
epochs = 200 # Количество проходов по данным
for epoch in range(epochs):
    optimizer.zero_grad()             # Обнуляем градиенты
    predictions = model(X_train_tensor) # Делаем предсказание
    loss = criterion(predictions, y_train_tensor) # Считаем ошибку
    loss.backward()                   # Считаем, как изменить веса
    optimizer.step()                  # Обновляем веса

    if (epoch+1) % 20 == 0:
        print(f"Эпоха {epoch+1}/{epochs} | Ошибка (Loss): {loss.item():.4f}")

# Проверяем на тестовой выборке
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_loss = criterion(test_preds, y_test_tensor)
print(f"\n✅ Обучение завершено. Ошибка на неизвестных данных: {test_loss.item():.4f}")

print("5. Сохранение модели и энкодеров...")
# Сохраняем веса модели
torch.save(model.state_dict(), "yield_model.pth")
# Сохраняем скейлеры и словари (ОБЯЗАТЕЛЬНО для инференса в приложении!)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_area, "le_area.pkl")
joblib.dump(le_item, "le_item.pkl")

print(f"Всё готово! Созданы файлы: yield_model.pth, scaler.pkl, le_area.pkl, le_item.pkl")
