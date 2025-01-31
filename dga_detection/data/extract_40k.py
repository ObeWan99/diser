import pandas as pd

# Замените "tranco_Z38ZG.csv" на ваш файл
input_file = "tranco_Z38ZG.csv"
output_file = "legit_dns.csv"

# Читаем только первые 40 000 строк
df = pd.read_csv(input_file, nrows=40000)

# Сохраняем новый файл
df.to_csv(output_file, index=False)

print(f"Файл {output_file} создан с первыми 40 000 строками.")