# Sentiment Analysis on Tweets

A deep learning micro project that classifies tweets as **positive** or **negative** using a
Bidirectional LSTM, with a Streamlit web app for live predictions.

## Dataset
[Sentiment140](http://help.sentiment140.com/) — 1.6 million tweets labelled positive/negative.
The training script uses a balanced random sample of 200,000 tweets (80/10/10 train/val/test split),
and also reports accuracy on the dataset's separate hand-labelled test set.

## Model
```
raw tweet → clean_text() → TextVectorization (20k vocab, 50 tokens)
          → Embedding (128) → Bidirectional LSTM (64) → Dropout
          → Dense (64, ReLU) → Dropout → Dense (1, sigmoid)
```
Loss: binary cross-entropy · Optimizer: Adam · Early stopping on validation loss.

## Project structure
```
├── data/                 Sentiment140 CSVs
├── models/               saved model, metrics.json, training_history.png
├── utils.py              text cleaning (URLs, @mentions, hashtags, repeated letters)
├── train.py              training + evaluation
├── app.py                Streamlit web app
└── requirements.txt
```

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# download the dataset into data/
curl -L -o data/s140.zip https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip data/s140.zip -d data && rm data/s140.zip

python train.py            # optional: the trained model is already in models/
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** and choose this repository, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose Python **3.12**, then deploy.

The trained model (`models/sentiment_model.keras`, ~11 MB) is committed, so the app runs without
the dataset or retraining.

## Web app features
- **Analyze a tweet** — type text, get Positive/Negative with confidence
- **Batch (CSV)** — upload a CSV with a `text` column, see the sentiment distribution, download results
- **Model info** — accuracy metrics and training curves
