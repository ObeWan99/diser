# Описание используемых библиотек

## pandas
**Описание:**  
Библиотека для работы с табличными данными. Позволяет загружать, обрабатывать и анализировать данные.

**Установка:**
```bash
pip install pandas
```

**Пример использования:**
```python
import pandas as pd

df = pd.read_csv("dataset.csv")
print(df.head())
```

---

## numpy
**Описание:**  
Библиотека для работы с многомерными массивами и выполнения числовых вычислений.

**Установка:**
```bash
pip install numpy
```

**Пример использования:**
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(np.mean(arr))
```

---

## re (Регулярные выражения)
**Описание:**  
Модуль для работы с текстами и поиска шаблонов.

**Пример использования:**
```python
import re

text = "DGA domains: abc123.com, xyz987.net"
pattern = r"[a-z0-9]+\.com"

matches = re.findall(pattern, text)
print(matches)  # ['abc123.com']
```

---

## scikit-learn
**Описание:**  
Библиотека для машинного обучения, включающая алгоритмы классификации, регрессии и кластеризации.

**Установка:**
```bash
pip install scikit-learn
```

**Пример использования:**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.rand(100, 5)
y = np.random.randint(2, size=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)
```

---

## matplotlib
**Описание:**  
Библиотека для визуализации данных.

**Установка:**
```bash
pip install matplotlib
```

**Пример использования:**
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 50]

plt.plot(x, y)
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
plt.title("График зависимости")
plt.show()
```

---

## seaborn
**Описание:**  
Библиотека для статистической визуализации данных.

**Установка:**
```bash
pip install seaborn
```

**Пример использования:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
sns.histplot(tips["total_bill"], kde=True)

plt.title("Распределение счетов")
plt.show()
```

---

## tqdm
**Описание:**  
Библиотека для создания индикаторов прогресса.

**Установка:**
```bash
pip install tqdm
```

**Пример использования:**
```python
from tqdm import tqdm
import time

for i in tqdm(range(100)):
    time.sleep(0.01)
```

# Установка окружения и запуск

```sh
# Установка Python 3.12
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# Создание нового окружения
python3 -m venv venv

# Активация виртуального окружения
source venv/bin/activate

# Установка зависимостей из файла requirements.txt
pip install -r requirements.txt

# Деактивация виртуального окружения
deactivate
```

```sh
# Загружаем и устанавливаем pyenv
curl https://pyenv.run | bash

# Добавляем переменную окружения PYENV_ROOT (указатель на папку, где установлен pyenv)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc

# Добавляем путь к pyenv в переменную PATH, если pyenv еще не добавлен
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc

# Инициализируем pyenv, чтобы он корректно работал в терминале
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Перезапускаем текущую оболочку (или можно просто открыть новый терминал)
exec "$SHELL"

# Устанавливаем Python версии 3.11.4 с помощью pyenv
pyenv install 3.11.4

# Переходим в каталог проекта (замените /path/to/project на нужный путь)
cd /path/to/project

# Устанавливаем локальную версию Python 3.11.4 для текущего каталога
pyenv local 3.11.4
```

# Дата сеты

1. DGA Domains Dataset на GitHub:
```sh
git clone https://github.com/chrmor/DGA_domains_dataset
```

2. DGA Dataset на Kaggle