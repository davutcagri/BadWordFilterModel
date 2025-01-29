import re
import nltk
from nltk import word_tokenize
from nltk.data import find
from nltk.corpus import stopwords

def clean(text):

    resources = ['tokenizers/punkt_tab', 'corpora/stopwords']
    missing_resources = []

    for resource in resources:
        try:
            find(resource)
        except LookupError:
            missing_resources.append(resource)

    if missing_resources:
        if missing_resources:
            print(f"Missing NLTK resources: {missing_resources}")
            print("Downloading missing resources...")
            for resource in missing_resources:
                nltk.download(resource.split('/')[-1])
            print("All NLTK resources are now available.")
        else:
            print("All required NLTK resources are already installed.")


    turkish_stopwords = set(stopwords.words('turkish'))

    text = text.lower() # Tüm harfleri küçük yap
    text = re.sub(r"[^\w\s]", "", text) # Noktalama ve özel karakterleri kaldır

    # Stopwordleri temizle
    texts = word_tokenize(text)
    cleanedTexts = [text for text in texts if text not in turkish_stopwords]
    return " ".join(cleanedTexts)