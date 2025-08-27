import pandas as pd
import nltk
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# NLTK ayarları
nltk.download('stopwords')
from nltk.corpus import stopwords

# 📌 1. Veri Yükleme
df = pd.read_csv("IMDB Dataset.csv")

# 📌 2. Özellik & Etiket
X = df['review']
y = df['sentiment']

# 📌 3. TF-IDF Vektörleştirme
tfidf = TfidfVectorizer(stop_words=stopwords.words("english"), max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# 📌 4. Eğitim / Test bölme
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 📌 5. Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 📌 Tkinter Arayüz
root = tk.Tk()
root.title("IMDB Sentiment Analysis")
root.geometry("500x380")

# Label
label = tk.Label(root, text="Bir film yorumu giriniz:", font=("Arial", 12))
label.pack(pady=10)

# Textbox
text_entry = tk.Text(root, height=6, width=50)
text_entry.pack(pady=5)

# Accuracy Label
accuracy_label = tk.Label(root, text="", font=("Arial", 10), fg="green")
accuracy_label.pack(pady=10)

# Tahmin fonksiyonu
def predict_sentiment():
    review = text_entry.get("1.0", tk.END).strip()
    if review:
        review_tfidf = tfidf.transform([review])
        prediction = model.predict(review_tfidf)[0]
        # Tahmin mesajı
        messagebox.showinfo("Tahmin Sonucu", f"Yorumunuz {prediction.upper()} olarak tahmin edildi ✅")
        # Accuracy güncelleme
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        accuracy_label.config(text=f"Eğitim Doğruluğu: {train_acc*100:.2f}% | Test Doğruluğu: {test_acc*100:.2f}%")
    else:
        messagebox.showwarning("Uyarı", "Lütfen bir yorum giriniz!")

# Button
predict_button = tk.Button(root, text="Tahmin Et", command=predict_sentiment, bg="blue", fg="white")
predict_button.pack(pady=10)

# Run
root.mainloop()
