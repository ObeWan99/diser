import pandas as pd
import os

# Указываем входной и выходной файлы
input_file = os.path.join("..", "data", "DGA_domains_dataset", "dga_domains_sample.csv")  # Укажите путь к вашему CSV
output_file = os.path.join("..", "domains.csv")

# Читаем CSV, добавляем заголовки
df = pd.read_csv(input_file, header=None, names=["label", "type", "domain"])

# Кодируем метки: 0 - легитимные (legit), 1 - DGA
df["label"] = df["label"].map({"legit": 0, "dga": 1})

# Проверяем, нет ли ошибок в данных
print(df.head())

# Сохраняем файл в корректном формате
df.to_csv(output_file, index=False)

print(f"Файл сохранён как {output_file}")
