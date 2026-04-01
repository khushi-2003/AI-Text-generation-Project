import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Text Generation Project",
    page_icon="📝"
)

# Yellow background
st.markdown(
    """
    <style>
    body {
        background-color: #ffff99;  /* Light yellow */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Text Generator by Khushi Kanade")

# ----------------------------
# Load model & tokenizer
# ----------------------------
model = load_model("TextGenModel.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_length = 5

# Create index -> word mapping to fix NameError
index_word = {v: k for k, v in tokenizer.word_index.items()}

# ----------------------------
# Temperature sampling 
# ----------------------------
def sample_with_temp(preds, temperature=0.7, top_k=4):
    preds = np.asarray(preds).astype("float64")

    # Selecting top k probabilities
    top_indices = np.argsort(preds)[-top_k:]
    top_probs = preds[top_indices]

    # Applying temperature scaling
    top_probs = np.log(top_probs + 1e-10) / temperature
    exp_probs = np.exp(top_probs)
    top_probs = exp_probs / np.sum(exp_probs)

    return np.random.choice(top_indices, p=top_probs)

# ----------------------------
# Text generation function
# ----------------------------
def gen_text(seed_text, next_words=20):
    output_text = seed_text
    generated_words = []

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_length - 1,
            padding='pre'
        ) 

        predicted_probs = model.predict(token_list, verbose=0)[0] 

        predicted_index = sample_with_temp(
            predicted_probs,
            temperature=0.8,
            top_k=5
        )

        nxt = index_word.get(predicted_index, "")

        # Avoid repeating the same word in last 5 words
        if nxt in generated_words[-5:]:
            continue

        generated_words.append(nxt)
        output_text += " " + nxt

    return output_text

# ----------------------------
# Streamlit UI
# ----------------------------
seed_text = st.text_input("Enter seed text:")

if st.button("Generate"):
    if seed_text:
        result = gen_text(seed_text)
        st.write(result)
    else:
        st.warning("Please enter some text")