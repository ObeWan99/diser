import joblib

# Загрузка модели и TF-IDF векторизатора
model = joblib.load("dga_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_domain(domain):
    domain_vector = vectorizer.transform([domain])  # Преобразуем домен в вектор
    prediction = model.predict(domain_vector)[0]  # Делаем предсказание
    probability = model.predict_proba(domain_vector)[0][1]  # Вероятность DGA

    result = "DGA" if prediction == 1 else "Legit"
    print(f"\nДомен: {domain}")
    print(f"Классификация: {result}")
    print(f"DGA Score: {probability:.4f}")

# Бесконечный цикл для ручного ввода доменов
while True:
    domain = input("\nВведите домен (или 'exit' для выхода): ").strip()
    if domain.lower() == "exit":
        print("Выход из программы.")
        break
    predict_domain(domain)
