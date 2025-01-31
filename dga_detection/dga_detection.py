import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка данных
df = pd.read_csv("domains.csv")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df["domain"], df["label"], test_size=0.2, random_state=42)

# Преобразование доменов в числовые вектора (TF-IDF на символьных N-граммах)
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Обучение модели (логистическая регрессия)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Предсказания на тестовом наборе
y_pred = model.predict(X_test_tfidf)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")