# spam_classifier.py  (robust version)
"""
SpamShield-AI (robust)
- Tries to load a public SMS dataset from GitHub.
- If network/download fails, falls back to a small built-in sample dataset so the script still runs.
- Trains TF-IDF + MultinomialNB and prints accuracy + sample predictions.
"""

import sys

# --- check for required packages and give friendly errors ---
required = ["pandas", "sklearn"]
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except Exception:
        missing.append(pkg)
if missing:
    print("ERROR: Missing Python packages:", ", ".join(missing))
    print("Install them by running:")
    print("   pip install pandas scikit-learn")
    sys.exit(1)

# now safe to import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import urllib.request
import io

REMOTE_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

def load_remote_dataset(url):
    try:
        print("[INFO] Attempting to download dataset from:", url)
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read()
        df = pd.read_csv(io.BytesIO(raw), sep='\t', header=None, names=['label','message'])
        print(f"[INFO] Downloaded dataset rows: {len(df)}")
        return df
    except Exception as e:
        print("[WARN] Could not download remote dataset:", repr(e))
        return None

def builtin_fallback():
    print("[INFO] Using small built-in fallback dataset (for demo).")
    data = [
        ("ham", "Ok lar... Joking wif u oni"),
        ("ham", "I'll be there in 5 minutes"),
        ("spam", "WINNER! You have won a free ticket. Call now"),
        ("spam", "Congratulations! You've been selected for a prize"),
        ("ham", "Can you come to college today?"),
        ("spam", "Get a loan at 0% interest. Apply now"),
        ("ham", "Don't forget to bring the notes."),
        ("spam", "You have WON $1000. Claim here"),
    ]
    df = pd.DataFrame(data, columns=['label','message'])
    return df

def main():
    df = load_remote_dataset(REMOTE_URL)
    if df is None or df.shape[0] < 50:
        df = builtin_fallback()

    # Basic sanity checks
    if 'label' not in df.columns or 'message' not in df.columns:
        print("[ERROR] Dataset does not contain expected columns 'label' and 'message'.")
        sys.exit(1)

    # quick class balance info
    print("[INFO] Label counts:\n", df['label'].value_counts())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_t = tfidf.fit_transform(X_train)
    X_test_t = tfidf.transform(X_test)

    # Train
    model = MultinomialNB()
    model.fit(X_train_t, y_train)

    # Evaluate
    pred = model.predict(X_test_t)
    acc = accuracy_score(y_test, pred)
    print(f"[RESULT] Test accuracy: {acc:.4f} (n_test={len(y_test)})")

    # show a few predictions for manual check
    print("\nSample predictions:")
    for i, txt in enumerate(X_test[:5]):
        p = model.predict(tfidf.transform([txt]))[0]
        print(f"  - '{txt[:60]}' -> {p}")

if __name__ == "__main__":
    main()

