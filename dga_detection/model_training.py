import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка данных
df = pd.read_csv("domains.csv")

# Проверка баланса классов
print("Баланс классов:\n", df["label"].value_counts())

# Разделение на обучающую и тестовую выборки (с балансировкой)
X_train, X_test, y_train, y_test = train_test_split(df["domain"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# Преобразование доменов в числовые вектора (TF-IDF на N-граммах)
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Конвертируем разреженные (sparse) матрицы в плотные (dense) массивы
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

# Обучение модели (HistGradientBoostingClassifier)
model = HistGradientBoostingClassifier(max_iter=500)
model.fit(X_train_dense, y_train)

# Предсказания на тестовом наборе
y_pred = model.predict(X_test_dense)
y_pred_proba = model.predict_proba(X_test_dense)[:, 1]

# Оптимальный порог
threshold = 0.6
y_pred_adjusted = (y_pred_proba > threshold).astype(int)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred_adjusted)
precision = precision_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)
f1 = f1_score(y_test, y_pred_adjusted)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Кросс-валидация
cv_scores = cross_val_score(model, X_train_dense, y_train, cv=5, scoring="f1")
print(f"Средний F1 Score на кросс-валидации: {cv_scores.mean():.4f}")

# Сохранение модели и векторизатора
joblib.dump(model, "dga_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Модель и векторизатор сохранены")
