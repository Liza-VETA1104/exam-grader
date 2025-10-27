import pandas as pd
import numpy as np
import re
import joblib
from scipy.sparse import hstack, csr_matrix
import streamlit as st
import tempfile
import os

# Настройка страницы
st.set_page_config(page_title="Оценка экзамена", page_icon="📝", layout="wide")
st.title("📝 Автоматическая оценка устного экзамена по русскому языку")
st.write("Загрузи CSV файл с экзаменационными данными и получи оценки")

# === Загрузка моделей ===
@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        model_q1 = joblib.load('model_q1.pkl')
        model_q2 = joblib.load('model_q2.pkl')
        model_q3 = joblib.load('model_q3.pkl')
        model_q4 = joblib.load('model_q4_enhanced.pkl')
        tfidf_q4 = joblib.load('tfidf_q4.pkl')
        scaler_q4 = joblib.load('scaler_q4.pkl')
        return tfidf, scaler, model_q1, model_q2, model_q3, model_q4, tfidf_q4, scaler_q4
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {e}")
        return None

# Показываем загрузку моделей
with st.spinner('Загружаем модели...'):
    models = load_models()

if models is None:
    st.stop()

tfidf, scaler, model_q1, model_q2, model_q3, model_q4, tfidf_q4, scaler_q4 = models
st.success("✅ Модели загружены!")

# === Функции обработки (твои оригинальные функции) ===
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', str(text)).replace('  ', ' ').strip()

def remove_instruction(transcript, q_num):
    if q_num == 1: start_phrase = "Начинайте свой диалог."
    elif q_num == 2: start_phrase = "Ответьте на вопросы собеседника полными предложениями."
    elif q_num == 3: start_phrase = "Поблагодарите за предоставленную информацию."
    elif q_num == 4: start_phrase = "Когда будете готовы, можете начинать описывать."
    else: return transcript
    idx = transcript.find(start_phrase)
    return transcript[idx + len(start_phrase):].strip() if idx != -1 else transcript

def extract_features(text):
    sentences = re.split(r'[.!?]+', text)
    n_sents = len([s for s in sentences if len(s.strip()) > 0])
    words = text.split()
    n_words = len(words)
    avg_sent_len = n_words / n_sents if n_sents > 0 else 0
    return [n_sents, n_words, avg_sent_len, int('?' in text)]

def get_q4_features_enhanced(text):
    text_low = text.lower()
    return {
        'has_season': int(any(w in text_low for w in ['лето', 'зима', 'весна', 'осень', 'тёплое время', 'снег', 'дождь'])),
        'has_place': int(any(w in text_low for w in ['кухня', 'дом', 'парк', 'вокзал', 'река', 'улица'])),
        'has_people_count': int(any(w in text_low for w in ['один', 'два', 'три', 'четыре', 'много детей', 'целая семья'])),
        'has_family': int(any(w in text_low for w in ['в нашей семье', 'у меня трое детей', 'я старшая', 'мой брат'])),
        'has_hobby': int(any(w in text_low for w in ['люблю готовить', 'играю в футбол', 'гуляю на природе', 'вышиваю'])),
        'n_sentences': len(re.split(r'[.!?]+', text)),
        'is_structured': int(len(re.findall(r'\b(на картинке|изображено|я вижу|расскажу о)\b', text_low)) >= 1),
        'has_emotion': int(any(w in text_low for w in ['радостный', 'счастлив', 'улыбается', 'весело'])),
        'is_garbage': int(any(w in text_low for w in [
            'characterization', 'leather.ru', 'Feit', 'Паспортный канал', 'understanding'
        ]) or len(text.split()) < 3)
    }

# === Основная функция оценки ===
def grade_exam(uploaded_file):
    try:
        # Читаем файл
        df = pd.read_csv(uploaded_file, sep=';', on_bad_lines='skip')

        # Показываем прогресс
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔍 Проверяем данные...")
        progress_bar.progress(10)

        if 'Оценка экзаменатора' in df.columns:
            df = df.drop(columns=['Оценка экзаменатора'])

        required_cols = ['Id экзамена', 'Id вопроса', '№ вопроса', 'Текст вопроса',
                         'Картинка из вопроса', 'Транскрибация ответа', 'Ссылка на оригинальный файл запис']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Отсутствует колонка: {col}")

        status_text.text("🧹 Очищаем текст...")
        progress_bar.progress(30)

        df['Текст вопроса_clean'] = df['Текст вопроса'].apply(clean_html)
        df['cleaned_transcript'] = df.apply(
            lambda row: remove_instruction(row['Транскрибация ответа'], row['№ вопроса']),
            axis=1
        )
        df['combined_text'] = df['Текст вопроса_clean'] + ' [SEP] ' + df['cleaned_transcript'].fillna('')

        status_text.text("📊 Вычисляем оценки...")
        progress_bar.progress(60)

        y_pred = np.zeros(len(df), dtype=int)

        for q_num in [1, 2, 3, 4]:
            mask = df['№ вопроса'] == q_num
            if not mask.any():
                continue

            if q_num == 4:
                X_text = tfidf_q4.transform(df.loc[mask, 'combined_text'])
                ling_feat = np.array([extract_features(txt) for txt in df.loc[mask, 'cleaned_transcript'].fillna('')])
                ling_scaled = scaler_q4.transform(ling_feat)
                feats = df.loc[mask, 'cleaned_transcript'].apply(get_q4_features_enhanced)
                feature_cols = list(feats.iloc[0].keys())
                checklist_feat = np.array([list(f.values()) for f in feats])
                X = hstack([X_text, csr_matrix(ling_scaled), csr_matrix(checklist_feat)])
                pred_raw = model_q4.predict(X)
                pred_rounded = np.array([int(np.clip(round(p), 0, 2)) for p in pred_raw])
                pred_rounded[checklist_feat[:, -1] == 1] = 0
                y_pred[mask] = pred_rounded
            else:
                X_text = tfidf.transform(df.loc[mask, 'combined_text'])
                ling_feat = np.array([extract_features(txt) for txt in df.loc[mask, 'cleaned_transcript'].fillna('')])
                ling_scaled = scaler.transform(ling_feat)
                q_norm = np.full((mask.sum(), 1), q_num / 4.0)
                X = hstack([X_text, csr_matrix(q_norm), csr_matrix(ling_scaled)])
                model = {1: model_q1, 2: model_q2, 3: model_q3}[q_num]
                pred_raw = model.predict(X)
                if q_num in (1, 3):
                    pred_rounded = np.array([0 if p < 0.5 else 1 for p in pred_raw])
                else:
                    pred_rounded = np.array([int(np.clip(round(p), 0, 2)) for p in pred_raw])
                y_pred[mask] = pred_rounded

        df['Оценка экзаменатора'] = y_pred
        output_cols = ['Id экзамена', 'Id вопроса', '№ вопроса', 'Текст вопроса',
                       'Картинка из вопроса', 'Оценка экзаменатора',
                       'Транскрибация ответа', 'Ссылка на оригинальный файл запис']
        df = df[output_cols]

        status_text.text("✅ Готово!")
        progress_bar.progress(100)

        return df

    except Exception as e:
        st.error(f"Ошибка обработки: {e}")
        return None

# === Интерфейс загрузки ===
st.header("📁 Загрузка файла")

uploaded_file = st.file_uploader(
    "Выбери CSV файл с разделителем ';'",
    type=['csv'],
    help="Файл должен содержать колонки: Id экзамена, Id вопроса, № вопроса, Текст вопроса, Транскрибация ответа и др."
)

if uploaded_file is not None:
    st.success(f"✅ Файл загружен: {uploaded_file.name}")

    if st.button("🚀 Начать оценку", type="primary"):
        with st.spinner('Обрабатываем данные...'):
            result_df = grade_exam(uploaded_file)

        if result_df is not None:
            st.success("Оценка завершена!")

            # Показываем статистику
            st.subheader("📈 Статистика оценок")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего ответов", len(result_df))
            with col2:
                st.metric("Средняя оценка", f"{result_df['Оценка экзаменатора'].mean():.2f}")
            with col3:
                st.metric("Макс оценка", result_df['Оценка экзаменатора'].max())

            # Показываем таблицу с результатами
            st.subheader("📊 Результаты")
            st.dataframe(result_df.head(10))

            # Скачивание результата
            st.subheader("📥 Скачать результат")
            csv = result_df.to_csv(index=False, sep=';')
            st.download_button(
                label="Скачать CSV с оценками",
                data=csv,
                file_name="graded_exam_results.csv",
                mime="text/csv"
            )

st.info("💡• Загрузите файл в формате CSV с разделителем ';'")
