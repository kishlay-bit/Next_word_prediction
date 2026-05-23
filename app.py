import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------------------------
# Styling
# ------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.pred-box {
    padding: 16px;
    border-radius: 10px;
    background-color: #262730;
    text-align: center;
    margin-top: 8px;
}
.pred-word {
    font-size: 22px;
    font-weight: 700;
    margin: 8px 0;
    color: #FFFFFF;
}
.pred-label {
    font-size: 12px;
    color: #A0A0A0;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.conf-text {
    font-size: 13px;
    color: #7EB8F7;
    margin-top: 4px;
}
.sentence-box {
    background-color: #1E1E2E;
    border-left: 3px solid #7EB8F7;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 16px;
    margin-bottom: 12px;
    word-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------
# Session State Init
# ------------------------------------------------
if "sentence" not in st.session_state:
    st.session_state.sentence = ""


# ------------------------------------------------
# Load Models
# ------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        lstm_model = load_model("lstm_next_word_model.keras")
    except Exception:
        lstm_model = None

    try:
        rnn_model = load_model("rnn_next_word_model.keras")
    except Exception:
        rnn_model = None

    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        st.error("tokenizer.pkl not found.")
        st.stop()

    try:
        with open("max_len.pkl", "rb") as f:
            max_len = pickle.load(f)
    except FileNotFoundError:
        st.error("max_len.pkl not found.")
        st.stop()

    index_to_word = {
        index: word
        for word, index in tokenizer.word_index.items()
    }

    return lstm_model, rnn_model, tokenizer, max_len, index_to_word


lstm_model, rnn_model, tokenizer, max_len, index_to_word = load_resources()


# ------------------------------------------------
# Prediction Function
# ------------------------------------------------
def predict_next_word(model, text):
    if model is None:
        return "Model not loaded", 0.0

    sequence = tokenizer.texts_to_sequences([text])[0]
    if len(sequence) == 0:
        return "Unknown input", 0.0

    sequence = pad_sequences([sequence], maxlen=max_len - 1, padding="pre")
    prediction = model.predict(sequence, verbose=0)
    predicted_index = int(np.argmax(prediction))
    predicted_word = index_to_word.get(predicted_index, "Not Found")
    confidence = float(np.max(prediction)) * 100

    return predicted_word, confidence


# ------------------------------------------------
# Append word callback — must happen BEFORE widgets render
# ------------------------------------------------
def append_word(word):
    st.session_state.sentence = st.session_state.sentence.strip() + " " + word.strip()


def reset():
    st.session_state.sentence = ""


# ------------------------------------------------
# UI
# ------------------------------------------------
st.title("🧠 Next Word Prediction")
st.caption("Type a sentence, then click a prediction to keep building it word by word.")

# Display the growing sentence (read-only, always reflects session state)
st.markdown("#### 📝 Current Sentence")
if st.session_state.sentence.strip():
    st.markdown(
        f'<div class="sentence-box">{st.session_state.sentence.strip()}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="sentence-box" style="color:#555;">Your sentence will appear here...</div>',
        unsafe_allow_html=True
    )

# Manual text input — updates sentence when user types
new_input = st.text_input(
    "Or type / edit directly here",
    value=st.session_state.sentence,
    placeholder="Deep learning models are"
)

# Sync manual edits back to session state
if new_input != st.session_state.sentence:
    st.session_state.sentence = new_input

st.button("🔄 Reset", on_click=reset)

# ------------------------------------------------
# Prediction Output
# ------------------------------------------------
if st.session_state.sentence.strip():
    with st.spinner("Predicting..."):
        lstm_word, lstm_conf = predict_next_word(lstm_model, st.session_state.sentence)
        rnn_word, rnn_conf = predict_next_word(rnn_model, st.session_state.sentence)

    st.markdown("#### Click a prediction to append it")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">LSTM Prediction</div>
                <div class="pred-word">{lstm_word}</div>
                <div class="conf-text">{lstm_conf:.2f}% confidence</div>
            </div>
        """, unsafe_allow_html=True)
        st.button(
            f'➕ "{lstm_word}"',
            key="use_lstm",
            use_container_width=True,
            on_click=append_word,
            args=(lstm_word,)
        )

    with col2:
        st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">RNN Prediction</div>
                <div class="pred-word">{rnn_word}</div>
                <div class="conf-text">{rnn_conf:.2f}% confidence</div>
            </div>
        """, unsafe_allow_html=True)
        st.button(
            f'➕ "{rnn_word}"',
            key="use_rnn",
            use_container_width=True,
            on_click=append_word,
            args=(rnn_word,)
        )

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.caption("LSTM & RNN Next Word Prediction · Built by Kishlay Tejeswi")
