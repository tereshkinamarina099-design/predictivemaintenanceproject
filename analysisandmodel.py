import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

def analysisandmodel_page():
    st.title("🔧 Анализ данных и предсказание отказов")
    
    st.write("### Загрузка данных")
    
    # Кнопка для загрузки данных
    if st.button("Загрузить данные из UCI"):
        with st.spinner("Загружаю данные..."):
            # Загрузка датасета
            dataset = fetch_ucirepo(id=601)
            data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            
            # Предобработка
            data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
            data['Type'] = LabelEncoder().fit_transform(data['Type'])
            
            # Масштабирование
            scaler = StandardScaler()
            numerical_features = ['Air temperature [K]', 'Process temperature [K]', 
                                  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
            data[numerical_features] = scaler.fit_transform(data[numerical_features])
            
            st.session_state['data'] = data
            st.success(f"✅ Данные загружены! {len(data)} записей")
            
            st.write("### Первые 5 строк данных:")
            st.dataframe(data.head())
    
    # Проверка, загружены ли данные
    if 'data' not in st.session_state:
        st.info("👆 Нажми кнопку выше, чтобы загрузить данные")
        return
    
    data = st.session_state['data']
    
    # Подготовка данных для обучения
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    st.write("### 🧠 Обучение модели")
    
    if st.button("Обучить модель Random Forest"):
        with st.spinner("Обучение модели..."):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            st.session_state['model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.success("✅ Модель обучена!")
    
    if 'model' not in st.session_state:
        st.info("👆 Нажми кнопку выше, чтобы обучить модель")
        return
    
    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Метрики
    st.write("### 📊 Результаты модели")
    
    col1, col2 = st.columns(2)
    with col1:
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Точность (Accuracy)", f"{accuracy:.2%}")
    
    with col2:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        st.metric("ROC-AUC", f"{roc_auc:.3f}")
    
    # Матрица ошибок
    st.write("#### Матрица ошибок (Confusion Matrix)")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Реальность')
    st.pyplot(fig)
    
    # ROC-кривая
    st.write("#### ROC-кривая")
    fig2, ax2 = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', label='Случайное угадывание')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    st.pyplot(fig2)
    
    # Отчет о классификации
    st.write("#### Отчет о классификации")
    st.text(classification_report(y_test, y_pred))
    
    # Предсказание для новых данных
    st.write("### 🔮 Предсказание для нового оборудования")
    st.write("Введи параметры оборудования:")
    
    col1, col2 = st.columns(2)
    with col1:
        type_val = st.selectbox("Тип продукта", ["L", "M", "H"])
        air_temp = st.number_input("Температура воздуха (K)", value=300.0)
        process_temp = st.number_input("Температура процесса (K)", value=310.0)
    with col2:
        rot_speed = st.number_input("Скорость вращения (rpm)", value=1500)
        torque = st.number_input("Крутящий момент (Nm)", value=40.0)
        tool_wear = st.number_input("Износ инструмента (min)", value=100)
    
    if st.button("🔍 Предсказать отказ"):
        # Преобразование ввода
        type_num = {"L": 0, "M": 1, "H": 2}[type_val]
        
        input_data = pd.DataFrame([[
            type_num, air_temp, process_temp, rot_speed, torque, tool_wear
        ]], columns=['Type', 'Air temperature [K]', 'Process temperature [K]',
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
        
        # Масштабирование
        scaler = StandardScaler()
        numerical_features = ['Air temperature [K]', 'Process temperature [K]', 
                              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        input_data[numerical_features] = scaler.fit_transform(data[numerical_features])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            st.error(f"⚠️ ОТКАЗ! Вероятность отказа: {probability:.1%}")
        else:
            st.success(f"✅ Оборудование работает. Вероятность отказа: {probability:.1%}")

if __name__ == "__main__":
    analysisandmodel_page()