#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans


# In[2]:


df = pd.read_csv('job_salary_prediction_dataset.csv')


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")


# In[6]:


st.set_page_config(
    page_title="Зарплатний дашборд",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': 'https://github.com/streamlit/streamlit/issues',
        'About': 'Інтерактивна панель для аналізу рівню зарплат'
    }
)


# In[10]:


# Бічна панель / SIDEBAR
st.sidebar.title("Панель фільтрації")

# Фільтри
selected_year = st.sidebar.selectbox("Досвід в роках", sorted(df["experience_years"].unique()))
selected_region = st.sidebar.multiselect("Регіон", df["location"].unique(), default=df["location"].unique())
selected_industry = st.sidebar.multiselect("Галузь", df["industry"].unique(), default=df["industry"].unique())
selected_scenario = st.sidebar.radio("Сертифікація", df["certifications"].unique())

selected_max_adbudget = st.sidebar.slider(
    "Максимальна заробітна плата",
    min_value=int(df["salary"].min()),
    max_value=int(df["salary"].max()),
    value=int(df["salary"].max()),
    step=1000
)

# Чекбокси для відображення
show_map = st.sidebar.checkbox("Показати Зарплата vs Досвід")

# Перемикач графіків
chart_option = st.sidebar.radio(
    "📈 Оберіть графік для перегляду:",
    [
        "Зарплата vs Досвід",
        "Boxplot зарплати по галузях",
        "Scatter: Зарплата vs Кількість скілів",
        "Гістограма сертифікацій по галузях",
        "Теплова карта кореляцій",
        "Кластеризація кандидатів (KMeans)"
    ]
)

# Інформаційний блок
st.sidebar.markdown("---")
st.sidebar.markdown(" **Інструкція**: \nФільтруйте дані за параметрами і переглядайте графіки та таблиці на панелі праворуч.")


# In[11]:


df_filtered = df[
    (df["experience_years"] == selected_year) &
    (df["location"].isin(selected_region)) &
    (df["industry"].isin(selected_industry)) &
    (df["certifications"] == selected_scenario) &
    (df["salary"] <= selected_max_adbudget)
]


# In[12]:


# Блок регресії
st.sidebar.markdown("Побудова регресії")
numeric_columns = df_filtered.select_dtypes(include=np.number).columns.tolist()

reg_x = st.sidebar.selectbox("Оберіть змінну X", numeric_columns, index=0)
reg_y = st.sidebar.selectbox("Оберіть змінну Y", numeric_columns, index=1)
show_regression = st.sidebar.checkbox("Показати регресійну модель")


# In[13]:


# Основна панель
st.title("Зарплатний дашборд")
st.subheader(f"🔍 Відфільтровано {df_filtered.shape[0]} зарплат")

# Кнопка завантаження CSV
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Завантажити CSV",
    data=csv,
    file_name="filtered_salary.csv",
    mime="text/csv"
)

# Інтерактивна таблиця результатів
st.subheader("Оберіть, які стовпці таблиці відображати")

all_columns = df_filtered.columns.tolist()
default_columns = ["job_title", "experience_years", "industry", "location", "salary"]

selected_columns = st.multiselect(
    "Оберіть стовпці для перегляду",
    options=all_columns,
    default=[col for col in default_columns if col in all_columns]
)

if selected_columns:
    st.dataframe(df_filtered[selected_columns])
else:
    st.info("Оберіть хоча б один стовпець, щоб побачити таблицю.")

if chart_option == "Зарплата vs Досвід":
    st.subheader("📊 Зарплата vs Досвід")
    chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x="experience_years:Q",
        y="salary:Q",
        color="industry:N",
        tooltip=["job_title", "experience_years", "salary", "industry"]
    ).interactive().properties(title="Зарплата vs Досвід")
    st.altair_chart(chart, use_container_width=True)

elif chart_option == "Boxplot зарплати по галузях":
    st.subheader("📊 Boxplot зарплати по галузях")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_filtered, x="industry", y="salary", ax=ax)
    ax.set_title("Розподіл зарплат по галузях")
    st.pyplot(fig)

elif chart_option == "Scatter: Зарплата vs Кількість скілів":
    st.subheader("📊 Scatter: Зарплата vs Кількість скілів")
    chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x="skills_count:Q",
        y="salary:Q",
        color="industry:N",
        tooltip=["job_title", "skills_count", "salary"]
    ).interactive().properties(title="Зарплата vs Кількість скілів")
    st.altair_chart(chart, use_container_width=True)

elif chart_option == "Гістограма сертифікацій по галузях":
    st.subheader("📊 Гістограма сертифікацій по галузях")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df_filtered,
        x="industry",
        y="certifications",
        estimator="mean",
        ax=ax
    )
    ax.set_title("Середня кількість сертифікацій по галузях")
    st.pyplot(fig)

elif chart_option == "Теплова карта кореляцій":
    st.subheader("📊 Теплова карта кореляцій")
    numeric_cols = df_filtered.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Кореляційна матриця числових показників")
    st.pyplot(fig)

elif chart_option == "Кластеризація кандидатів (KMeans)":
    st.subheader("📊 Кластеризація кандидатів за зарплатою та досвідом")
    cluster_data = df_filtered[["salary", "experience_years"]].dropna().copy()

    if cluster_data.shape[0] >= 3:
        kmeans = KMeans(n_clusters=3, random_state=0)
        cluster_data["Cluster"] = kmeans.fit_predict(cluster_data)

        chart = alt.Chart(cluster_data).mark_circle(size=60).encode(
            x="experience_years:Q",
            y="salary:Q",
            color="Cluster:N",
            tooltip=["salary", "experience_years", "Cluster"]
        ).interactive().properties(title="Кластеризація за зарплатою та досвідом")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Недостатньо даних для кластеризації (потрібно ≥ 3 рядки).")


# In[ ]:


if show_regression:
    st.subheader("📈 Лінійна регресія")

    X = df_filtered[[reg_x]].dropna()
    y = df_filtered[reg_y].dropna()

    # вирівнюємо індекси
    data = pd.concat([X, y], axis=1).dropna()
    X = data[[reg_x]]
    y = data[reg_y]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Дані")
    ax.plot(X, y_pred, color="red", label="Регресія")
    ax.set_xlabel(reg_x)
    ax.set_ylabel(reg_y)
    ax.legend()

    st.pyplot(fig)

