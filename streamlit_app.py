import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Загрузка модели и токенизатора
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    return tokenizer, model

tokenizer, model = load_model()

# Функция для анализа тональности
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probabilities, dim=-1).item()
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return labels[sentiment], probabilities[0].tolist()

# Заголовок приложения
st.title("Анализ Тональности Текста")
st.write("Введите текст ниже, чтобы узнать его эмоциональную окраску!")

# Поле ввода текста
user_input = st.text_area("Введите текст:", value="", height=150)

if st.button("Анализировать"):
    if user_input.strip():
        sentiment, probabilities = analyze_sentiment(user_input)
        st.subheader("Результаты анализа:")
        st.write(f"**Тональность:** {sentiment}")
        st.write(f"**Вероятности:**")
        st.write(f"- Положительная: {probabilities[2]:.2f}")
        st.write(f"- Нейтральная: {probabilities[1]:.2f}")
        st.write(f"- Отрицательная: {probabilities[0]:.2f}")
    else:
        st.warning("Пожалуйста, введите текст для анализа.")

# Инструкция для запуска
st.markdown("---")
st.markdown("Запустите приложение командой: `streamlit run app.py`")

