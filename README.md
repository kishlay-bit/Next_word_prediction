# Next Word Prediction using LSTM

A deep learning NLP project that predicts the next word in a sequence using a Recurrent Neural Network (LSTM) trained on a quotes dataset.

---

## Overview

This project implements a next-word prediction system using Long Short-Term Memory (LSTM) networks. Given a sequence of words, the model predicts the most probable next word — similar to the autocomplete feature in keyboards and search engines.

---

## Demo

> Type a sequence of words → Model predicts the next word in real time via Flask API.

Example:
```
Input:  "the only way to"
Output: "do"

Input:  "in the middle of"
Output: "difficulty"
```

---

## Project Structure

```
Next_word_prediction/
│
├── RNNimplementation.ipynb   # Model architecture and training notebook
├── codefile.ipynb            # Data preprocessing and tokenization
├── app.py                    # Flask backend for serving predictions
├── lstm_model (1).h5         # Trained LSTM model weights
├── tokenizer.pkl             # Serialized tokenizer for inference
├── max_len.pkl               # Saved max sequence length for padding
└── qoute_dataset.csv         # Training dataset (quotes corpus)
```

---

## Model Architecture

| Component | Details |
|---|---|
| Model Type | Recurrent Neural Network (RNN) |
| Layer | LSTM (Long Short-Term Memory) |
| Framework | TensorFlow / Keras |
| Input | Tokenized and padded word sequences |
| Output | Softmax over vocabulary (next word probability) |
| Dataset | Quotes corpus (`qoute_dataset.csv`) |

**Pipeline:**
```
Raw Text → Tokenization → Sequence Generation → Padding → LSTM Training → Saved Model (.h5)
```

---

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **NLP:** Keras Tokenizer, Sequence Padding
- **Backend:** Flask
- **Serialization:** Pickle (tokenizer, max_len)
- **Environment:** Jupyter Notebook

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/kishlay-bit/Next_word_prediction.git
cd Next_word_prediction
```

### 2. Install Dependencies
```bash
pip install tensorflow flask numpy pandas scikit-learn
```

### 3. Run the Flask App
```bash
python app.py
```
The API will be available at `http://localhost:5000`

### 4. Make a Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "the only way to"}'
```

---

## How It Works

1. **Data Preprocessing** — Raw quotes are cleaned, tokenized, and converted into n-gram sequences
2. **Sequence Padding** — All sequences are padded to uniform length (`max_len`)
3. **Model Training** — LSTM network is trained to predict the next token given preceding tokens
4. **Inference** — Input text is tokenized, padded, and passed through the saved model; the output token is decoded back to a word
5. **Serving** — Flask app loads the saved `.h5` model and `tokenizer.pkl` at startup and exposes a REST API endpoint

---

## Results

The model successfully learns sequential word patterns from the quotes corpus and generates contextually relevant next-word predictions, demonstrating the effectiveness of LSTM networks for language modeling tasks.

---

## Future Improvements

- [ ] Add a frontend UI (React / Streamlit)
- [ ] Train on a larger, more diverse corpus
- [ ] Experiment with Bidirectional LSTM and Transformer architecture
- [ ] Deploy on Render / Railway for public access
- [ ] Add top-k predictions with confidence scores

---

## Author

**Kishlay Tejeswi**  
B.Tech CSE, BIT Mesra | Research Intern, IIT Kharagpur  
[LinkedIn](https://www.linkedin.com/in/kishlay-tejeswi/) • [GitHub](https://github.com/kishlay-bit)
