import ssl
import preview
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


ssl._create_default_https_context = ssl._create_unverified_context

path = '/Users/davutcagri/Documents/Projeler/BadWordFilterModel/datasets/train.csv'

df = pd.read_csv(path)

df = df[['text', 'label']] # Sadece text ve label sütunlarını seç
df['cleaned_text'] = df['text'].apply(preview.clean) # cleaned_text sütunu oluştur ve clean metodundan geçmiş text datalarını sütuna yaz

model = LogisticRegression()
tfidf = TfidfVectorizer()

# TF-IDF Vektörleştirme
X = tfidf.fit_transform(df['cleaned_text'])
Y = df['label']

#Tüm veriyi kullanarak modeli eğitmek
model.fit(X, Y)

def test():
    test_df = pd.read_csv('/Users/davutcagri/Documents/Projeler/BadWordFilterModel/datasets/test.csv')
    X_test = tfidf.transform(test_df['text'])
    Y_test = test_df['label']

    y_pred = model.predict(X_test)

    print(classification_report(y_pred, Y_test))

def input_filter(input_text):
    X_input = tfidf.transform([input_text])
    prediction = model.predict(X_input)

    return prediction