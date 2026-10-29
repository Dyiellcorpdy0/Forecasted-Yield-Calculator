import pandas as pd

print("1. Читаем исходный датасет...")
df = pd.read_csv("yield_df.csv")
original_size = len(df)

# 1. Оставляем только нужные культуры
target_crops = ['Potatoes', 'Wheat']

# 2. Оставляем страны с климатом, похожим на Беларусь
target_countries = [
    'Belarus', 'Ukraine', 'Poland', 'Lithuania', 'Latvia', 'Estonia', 
    'Russian Federation', 'Germany', 'Czechia', 'Slovakia', 
    'Finland', 'Sweden', 'Canada', 'Norway', 'Austria', 'Romania'
]

print("2. Фильтруем данные...")
# Применяем фильтры
# Оставляем только те строки, где Культура есть в нашем списке И Страна есть в нашем списке
df_cleaned = df[df['Item'].isin(target_crops) & df['Area'].isin(target_countries)]

print("3. Сохраняем результат...")
# Сохраняем в новый файл, убрав колонку индексов, если она сгенерировалась
df_cleaned.to_csv("yield_df_cleaned.csv", index=False)

# Выводим статистику для проверки
print("\n=== ОТЧЕТ ОБ ОЧИСТКЕ ===")
print(f"Было строк: {original_size}")
print(f"Осталось строк: {len(df_cleaned)}")
print("-" * 25)
print(f"Оставшиеся культуры: {df_cleaned['Item'].unique().tolist()}")
print(f"Оставшиеся страны: {df_cleaned['Area'].unique().tolist()}")
print("========================")
print("✅ Готово! Создан файл: yield_df_cleaned.csv")
