# 🛫 Análisis de Tweets Negativos de Aerolíneas con Machine Learning y NLP

## 📌 Introducción
El análisis de sentimientos en redes sociales es una herramienta clave para las empresas del sector aéreo. En este proyecto analizamos el dataset **US Airlines Sentiment**, compuesto por **14,640 tweets** dirigidos a distintas aerolíneas en EE. UU.

El objetivo es **detectar patrones, anomalías y generar insights de negocio** usando técnicas de:

- Limpieza y exploración de datos (EDA).
- Procesamiento de Lenguaje Natural (NLP).
- Modelos supervisados y no supervisados de Machine Learning.
- Detección de anomalías con métodos estadísticos y de bosque aleatorio.

---

## 🧹 1. Preparación y Limpieza de Datos

### 🔹 Carga inicial del dataset:
```python
import pandas as pd

df = pd.read_csv("Tweets.csv", parse_dates=["tweet_created"], encoding="utf-8")
df.shape
```

### 🔹 Eliminación de columnas irrelevantes y gestión de valores nulos.

### 🔹 Creación de un dataset enfocado en tweets negativos:
```python
df_tweets_negativos = df[df['airline_sentiment'] == 'negative'].copy()
df_tweets_negativos.to_csv("tweets_negativos_limpios.csv", index=False)
```

---

## 📊 2. Exploración de Datos

### 🔹 Análisis Unidimensional
- **Variables numéricas:** distribución de `retweet_count`.
- **Variables categóricas:** frecuencia de aerolíneas.

```python
df_tweets_negativos['airline'].value_counts().plot(kind="bar", figsize=(8,4))
```

📌 **Resultado:** *United* y *US Airways* concentran la mayor cantidad de quejas.

### 🔹 Análisis Bidimensional
Relación entre aerolínea y causa negativa (`negativereason`):

```python
pd.crosstab(df_tweets_negativos['airline'], df_tweets_negativos['negativereason'])
```

📌 **Insight:** Los retrasos y cancelaciones son las quejas más frecuentes.

---

## 📝 3. Análisis de Texto (NLP)

### 🔹 Tokenización y limpieza:
```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

tokens = word_tokenize(" ".join(df_tweets_negativos['text']).lower())
clean_tokens = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')]
```

### 🔹 Nube de palabras:
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(clean_tokens))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
```

📌 **Insight:** Palabras más comunes → *delay, cancelled, service, flight, customer*.

---

## 🤖 4. Modelos de Machine Learning

### 🔹 Modelo Supervisado (Clasificación)
Entrenamos un modelo **Regresión Logística** para predecir sentimiento.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['airline_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
```

### 🔹 Modelo No Supervisado (Clustering con K-Means)
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

df_tweets_negativos['cluster'] = labels
```

📌 **Insight:** Los clusters muestran grupos de quejas por **retrasos, mal servicio al cliente y cancelaciones**.

---

## 🚨 5. Detección de Anomalías

### 🔹 Isolation Forest
```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
df_tweets_negativos['anomaly_iso'] = iso.fit_predict(X.toarray())
```

### 🔹 Modelo Gaussiano (Elliptic Envelope)
```python
from sklearn.covariance import EllipticEnvelope

ee = EllipticEnvelope(contamination=0.05, random_state=42)
df_tweets_negativos['anomaly_gauss'] = ee.fit_predict(X.toarray())
```

📌 **Insight:** Los tweets anómalos tienden a usar **lenguaje extremo o poco frecuente**, detectando casos únicos de frustración o incidentes.

---

## 📈 6. Visualizaciones Destacadas
- 📊 **Gráfica de barras** → aerolíneas con más tweets negativos.  
- ☁️ **Nube de palabras** → términos más frecuentes en quejas.  
- 🔀 **WordCloud por cluster** → diferencias en el lenguaje de cada grupo.  
- 🚨 **Tweets anómalos vs normales** → comparación textual.  

---

## ✅ Conclusiones
1. **United y US Airways** concentran la mayor proporción de quejas.  
2. **Principales causas negativas**: retrasos, cancelaciones y mala atención.  
3. Los **clusters** revelan distintos focos de frustración de los clientes.  
4. La **detección de anomalías** permitió identificar mensajes únicos de alto impacto.  
5. Este flujo de trabajo demuestra cómo un **pipeline de NLP + ML + Análisis Estadístico** genera insights accionables para mejorar la experiencia de cliente en el sector aéreo.  

---

✍️ *Este análisis combina ciencia de datos, NLP y Machine Learning para transformar miles de tweets en insights que pueden guiar decisiones estratégicas en la industria aérea.*
