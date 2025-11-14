# SpamShield-AI: Simple SMS Spam Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset (public UCI dataset)
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                 sep='\t', header=None, names=['label','message'])

# Split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Convert text → numbers
tfidf = TfidfVectorizer(stop_words='english')
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Train classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Test
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)
