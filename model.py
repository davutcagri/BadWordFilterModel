import ssl
import preview
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


ssl._create_default_https_context = ssl._create_unverified_context

model = LogisticRegression()
tfidf = TfidfVectorizer()

train_path = 'datasets/train.csv'
valid_path = 'datasets/valid.csv'

df = pd.read_csv(train_path)
df = df[['text', 'label']] # Sadece text ve label sütunlarını seç
df['cleaned_text'] = df['text'].apply(preview.clean) # Ön temizlikten geçir

# TF-IDF Vektörleştirme
X_train = tfidf.fit_transform(df['cleaned_text'])
Y_train = df['label']

#Tüm veriyi kullanarak modeli eğitmek
model.fit(X_train, Y_train)

def classification_test():
    test_df = pd.read_csv('datasets/test.csv')

    X_test = tfidf.transform(test_df['text'])
    Y_test = test_df['label']

    Y_pred = model.predict(X_test)

    return classification_report(Y_pred, Y_test)

def input_filter(input_text):
    X_input = tfidf.transform([input_text])
    Y_pred = model.predict(X_input)

    return Y_pred