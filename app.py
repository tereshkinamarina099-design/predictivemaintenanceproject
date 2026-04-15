import streamlit as st

# Создаем страницы
pagina1 = st.Page("analysisandmodel.py", title="Анализ и модель")
pagina2 = st.Page("presentation.py", title="Презентация")

# Показываем навигацию
pg = st.navigation([pagina1, pagina2])
pg.run()