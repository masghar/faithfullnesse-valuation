# streamlit_app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import nltk

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from nltk.corpus import wordnet as wn

# -----------------------
# Streamlit config
# -----------------------
st.set_page_config(
    layout="wide",
    page_title="Faithfulness Evaluation (e-SNLI)"
)

# -----------------------
# NLTK downloads (required on Streamlit Cloud)
# -----------------------
@st.cache_resource
def setup_nltk():
    nltk.download("wordnet")
    nltk.download("omw-1.4")

setup_nltk()

# -----------------------
# Device (CPU only on Streamlit Cloud)
# -----------------------
@st.cache_resource
def get_device():
    return torch.device("cpu")

DEVICE = get_device()
st.sidebar.write(f"Using device: **{DEVICE}**")

# -----------------------
# Load model / tokenizer
# -----------------------
@st.cache_resource
def load_model_and_tokenizer(
    model_name="roberta-base-mnli"  # lighter + safer for Streamlit Cloud
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# -----------------------
# Load e-SNLI dataset
# -----------------------
@st.cache_resource
def load_esnli(n=200):
    df = pd.read_csv("data/esnli_dev.csv")

    df = df.rename(columns={
        "Sentence1": "premise",
        "Sentence2": "hypothesis",
        "gold_label": "label",
        "Explanation_1": "explanation"
    })

    label_map = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2
    }
    df["label"] = df["label"].map(label_map)

    df = df.dropna(subset=["premise", "hypothesis", "label", "explanation"])

    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.select(range(min(n, len(ds))))

esnli = load_esnli(n=200)

# -----------------------
# Token importance via gradients
# -----------------------
def compute_token_importance(premise, hypothesis):
    encoded = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    embedding_layer = model.get_input_embeddings()
    input_embeds = embedding_layer(input_ids)
    input_embeds = input_embeds.clone().detach().requires_grad_(True)

    outputs = model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask
    )

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    pred_label = torch.argmax(probs, dim=-1).item()
    pred_prob = probs[0, pred_label].item()

    model.zero_grad()
    logits[0, pred_label].backward()

    grads = input_embeds.grad
    importance = grads.abs().sum(dim=-1).squeeze(0).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0).cpu().tolist()
    )

    return tokens, importance, pred_label, pred_prob, encoded

# -----------------------
# Mask utility
# -----------------------
def mask_input_ids(encoded, mask_positions, mode="unk"):
    input_ids = encoded["input_ids"][0].cpu().tolist()

    new_ids = input_ids.copy()
    for pos in mask_positions:
        if pos >= len(new_ids):
            continue
        if mode == "mask" and tokenizer.mask_token_id is not None:
            new_ids[pos] = tokenizer.mask_token_id
        else:
            new_ids[pos] = tokenizer.unk_token_id

    return torch.tensor([new_ids])

def run_model(input_ids):
    input_ids = input_ids.to(DEVICE)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.softmax(outputs.logits, dim=-1)

    return probs.cpu().numpy()[0], int(np.argmax(probs.cpu().numpy()))

# -----------------------
# Token deletion test
# -----------------------
def token_deletion_test(encoded, importance, steps=8):
    idx_sorted = np.argsort(-importance)

    base_probs, base_pred = run_model(encoded["input_ids"])
    base_prob = base_probs[base_pred]

    results = []
    for k in range(1, steps + 1):
        ids = mask_input_ids(encoded, idx_sorted[:k])
        probs, pred = run_model(ids)
        results.append({
            "k": k,
            "prob": probs[base_pred],
            "pred": pred
        })

    return base_prob, results

# -----------------------
# Counterfactual substitution
# -----------------------
def find_antonym(token):
    word = token.replace("Ġ", "").lower()
    synsets = wn.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                return ant.name()
    return None

def counterfactual_test(encoded, tokens, importance, top_k=1):
    idx_sorted = np.argsort(-importance)
    base_probs, base_pred = run_model(encoded["input_ids"])

    results = []
    for i in range(top_k):
        pos = idx_sorted[i]
        antonym = find_antonym(tokens[pos])
        if antonym is None:
            continue

        ant_ids = tokenizer(antonym)["input_ids"][1:-1]
        ids = encoded["input_ids"][0].cpu().tolist()
        ids[pos] = ant_ids[0]

        probs, pred = run_model(torch.tensor([ids]))
        results.append({
            "token": tokens[pos],
            "antonym": antonym,
            "new_pred": pred,
            "prob_orig_pred": probs[base_pred]
        })

    return base_probs[base_pred], results

# -----------------------
# UI
# -----------------------
st.title("Faithfulness Evaluation — e-SNLI")

col1, col2 = st.columns([1, 2])

with col1:
    idx = st.number_input(
        "Sample index",
        0,
        len(esnli) - 1,
        0
    )

    test_type = st.selectbox(
        "Test",
        ["All", "Token Deletion", "Counterfactual"]
    )

    steps = st.slider("Deletion steps", 1, 20, 8)
    top_k = st.slider("Counterfactual top-k", 1, 5, 1)

    run = st.button("Run")

with col2:
    ex = esnli[int(idx)]
    st.markdown(f"**Premise:** {ex['premise']}")
    st.markdown(f"**Hypothesis:** {ex['hypothesis']}")
    st.markdown(f"**Gold label:** {ex['label']}")
    st.info(ex["explanation"])

# -----------------------
# Run
# -----------------------
if run:
    st.info("Computing gradients…")

    t0 = time.time()
    tokens, importance, pred, prob, encoded = compute_token_importance(
        ex["premise"],
        ex["hypothesis"]
    )
    st.success(f"Predicted label: {pred} | prob={prob:.4f}")

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.bar(range(len(importance)), importance)
    ax.set_title("Token importance (gradient saliency)")
    st.pyplot(fig)

    if test_type in ("All", "Token Deletion"):
        st.header("Token Deletion")
        base, res = token_deletion_test(encoded, importance, steps)
        st.write("Baseline prob:", base)
        st.table(res)

    if test_type in ("All", "Counterfactual"):
        st.header("Counterfactual Substitution")
        base, res = counterfactual_test(encoded, tokens, importance, top_k)
        st.write("Baseline prob:", base)
        st.table(res)
