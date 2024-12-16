# %% [markdown]
# # Imports/Setup

# %%
# %%bash

# if [[ ! -d "./data" ]]
# then
#   echo "Downloading files if missing"
#   git clone https://github.com/kabirahuja2431/CSE447-547MAutumn2024.git
#   cp -r ./CSE447-547MAutumn2024/"Project 2"/data .
#   cp ./CSE447-547MAutumn2024/"Project 2"/wordvec_tests.py .
#   cp ./CSE447-547MAutumn2024/"Project 2"/nn_tests.py .
#   cp ./CSE447-547MAutumn2024/"Project 2"/glove.py .
#   cp ./CSE447-547MAutumn2024/"Project 2"/siqa.py .
#   wget https://homes.cs.washington.edu/~kahuja/cse447/project2/glove.6B.50d.txt -O data/embeddings/glove.6B/glove.6B.50d.txt
#   wget https://homes.cs.washington.edu/~kahuja/cse447/project2/X_train_st.pt -O data/sst/X_train_st.pt
#   wget https://homes.cs.washington.edu/~kahuja/cse447/project2/X_dev_st.pt -O data/sst/X_dev_st.pt
#   wget https://homes.cs.washington.edu/~kahuja/cse447/project2/train_data_embedded.pt -O data/socialiqa-train-dev/train_data_embedded.pt
#   wget https://homes.cs.washington.edu/~kahuja/cse447/project2/dev_data_embedded.pt -O data/socialiqa-train-dev/dev_data_embedded.pt
# fi

# # %%
# %%bash
# # Install required packages
# pip install pandas
# pip install sentence-transformers
# pip install tf_keras

# %%
import os
import re
from typing import List, Dict
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np

from sentence_transformers import SentenceTransformer
nltk.download("punkt")
nltk.download('punkt_tab')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

from itertools import product

# %%
parent_dir = os.path.dirname(os.path.abspath("__file__"))
data_dir = os.path.join(parent_dir, "data")
wnli_dir = os.path.join(parent_dir, "WNLI")

# %% [markdown]
# # Load Data

# %% [markdown]
# ### WNLI Dataset

# %%
train_df = pd.read_csv(f'{wnli_dir}/train.tsv', sep='\t')
dev_df = pd.read_csv(f'{wnli_dir}/dev.tsv', sep='\t')

# %% [markdown]
# ### Glove Embeddings

# %%
class GloveEmbeddings:

    def __init__(self, path="embeddings/glove.6B/glove.6B.50d.txt"):
        """
        Initializes GloveEmbeddings object.

        Inputs:
        - path: The path to the GloVe embedding data
        
        """
        self.path = path
        self.vec_size = int(re.search(r"\d+(?=d)", path).group(0))
        self.embeddings = {}
        self.load()

    def load(self):
        """
        Loads the GloVe embedding data.
        
        """
        for line in open(self.path, "r"):
            values = line.split()

            word_len = len(values) - self.vec_size

            word = " ".join(values[:word_len])
            vector_values = list(map(float, values[word_len:]))

            word = values[0]
            vector_values = list(map(float, values[-self.vec_size :]))
            vector = torch.tensor(vector_values, dtype=torch.float)
            self.embeddings[word] = vector

    def is_word_in_embeddings(self, word):
        """
        Inputs:
        - word: The word to search for

        Returns:
        - bool: True if word is in the GloVe embedding data, false otherwise
        
        """
        return word in self.embeddings

    def get_vector(self, word):
        if not self.is_word_in_embeddings(word):
            return self.embeddings["unk"]
        return self.embeddings[word]

    def __getitem__(self, word):
        return self.get_vector(word)

glove_embeddings = GloveEmbeddings(
    path=f"{data_dir}/embeddings/glove.6B/glove.6B.50d.txt"
)

# %% [markdown]
# Defining sentence embedding function for GloveEmbeddings:

# %%
def get_sentence_embedding(
    sentence: str,
    word_embeddings: GloveEmbeddings,
    use_POS: bool = False,
    pos_weights: Dict[str, float] = None
):
    """
    Compute the sentence embedding using the word embeddings.

    Inputs:
    - sentence: The input sentence
    - word_embeddings: GloveEmbeddings object
    - use_POS: Whether to use POS tagging
    - pos_weights: Dictionary containing POS weights

    Returns:
    torch.Tensor: The sentence embedding
    """
    tokens = word_tokenize(sentence.lower())

    if use_POS:
        tags = nltk.pos_tag(tokens)
        embeddings = []
        for w, t in tags:
            emb = torch.zeros(word_embeddings.vec_size)
            if word_embeddings.is_word_in_embeddings(w) and t in pos_weights:
                emb = word_embeddings[w] * pos_weights[t]
            embeddings.append(emb)
    else:
        embeddings = [word_embeddings[w] for w in tokens if word_embeddings.is_word_in_embeddings(w)]

    if embeddings:
        return torch.sum(torch.stack(embeddings), dim=0)
    else:
        return torch.zeros((word_embeddings.vec_size))

# %% [markdown]
# ### Sentence Transformer

# %% [markdown]
# Defining sentence embedding function for SentenceTransformer:

# %%
def get_st_embeddings(
    sentences: List[str],
    st_model: SentenceTransformer,
    batch_size: int = 32,
    device: str = "cpu",
):
    """
    Compute the sentence embedding using the Sentence Transformer model.

    Inputs:
    - sentence: The input sentence
    - st_model: SentenceTransformer model
    - batch_size: Encode in batches to avoid memory issues in case multiple sentences are passed

    Returns:
    torch.Tensor: The sentence embedding of shape [d,] (when only 1 sentence) or [n, d] where n is the number of sentences and d is the embedding dimension
    """

    st_model.to(device)
    sentence_embeddings = None

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i : i + batch_size]
        batch_embeddings = st_model.encode(batch_sentences, convert_to_tensor=True)
        if sentence_embeddings is None:
            sentence_embeddings = batch_embeddings
        else:
            sentence_embeddings = torch.cat(
                [sentence_embeddings, batch_embeddings], dim=0
            )

    return sentence_embeddings.to("cpu")

# %% [markdown]
# ### Embed WNLI Data

# %%
def preprocess_wnli(embed_method, df):
    """
    Preprocesses data.

    Inputs:
    - embed_method: Either "glove" to use GloVe or "st" to use Sentence Transformers
    - df: Pandas dataframe

    Returns:
    List[Dict[str, torch.Tensor]]: Embeddings specified by embed_method
    """
    if embed_method == "glove":
        s1 = torch.stack([get_sentence_embedding(s, glove_embeddings, use_POS=False) 
                                for s in df["sentence1"].values])
        s2 = torch.stack([get_sentence_embedding(s, glove_embeddings, use_POS=False) 
                                for s in df["sentence2"].values])
    elif embed_method == "st":
        st_model = SentenceTransformer("all-mpnet-base-v2")
        s1 = get_st_embeddings(df["sentence1"].values, st_model, device=DEVICE)
        s2 = get_st_embeddings(df["sentence2"].values, st_model, device=DEVICE)

    return [{"sentence1": a, "sentence2": b} for a, b in zip(s1, s2)]

# %%
X_train_glove = preprocess_wnli("glove", train_df)
X_dev_glove = preprocess_wnli("glove", dev_df)
X_train_st = preprocess_wnli("st", train_df)
X_dev_st = preprocess_wnli("st", dev_df)

Y_train = torch.Tensor(train_df["label"].values)
Y_dev = torch.Tensor(dev_df["label"].values)

# %% [markdown]
# ### Dataloader

# %%
class WNLIEmbeddedDataset(torch.utils.data.Dataset):

    def __init__(self, embeddings: List[Dict[str, torch.Tensor]], labels: List[str]):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        sample = self.embeddings[idx]
        return {
            "sentence1": sample["sentence1"],
            "sentence2": sample["sentence2"],
            "label": self.labels[idx]
        }

def get_wnli_dataloader(
    embeddings: List[Dict[str, torch.Tensor]],
    labels: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
):

    dataset = WNLIEmbeddedDataset(embeddings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_glove_dataloader = get_wnli_dataloader(X_train_glove, Y_train, batch_size=32, shuffle=True)
train_st_dataloader = get_wnli_dataloader(X_train_st, Y_train, batch_size=32, shuffle=True)
dev_glove_dataloader = get_wnli_dataloader(X_dev_glove, Y_dev, batch_size=32, shuffle=False)
dev_glove_dataloader = get_wnli_dataloader(X_dev_st, Y_dev, batch_size=32, shuffle=False)

# %% [markdown]
# # Neural Network

# %%
class WNLIFFNN(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, depth = 1):
        super(WNLIFFNN, self).__init__()

        layers = [nn.Linear(input_dim * 2, hidden_dim), nn.ReLU()]

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        self.initialize_weights()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([s1, s2], dim=-1))

    def initialize_weights(self):
        for layer in self.modules():
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

# %%
def evaluate(
    model: WNLIFFNN,
    dev_data_embedded: List[Dict[str, torch.Tensor]],
    dev_labels: torch.Tensor,
    eval_batch_size: int = 128,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluates the model on the WNLI dataset.

    Inputs:
    - model: The WNLIFFNN model
    - dev_data_embedded: List of dictionaries containing the embedded context and disambiguation for the validation data
    - dev_labels: List of labels for the validation data
    - eval_batch_size: Batch size for evaluation
    - device: Device to run the evaluation on
    """
    model.eval()
    model.to(device)

    loader = get_wnli_dataloader(dev_data_embedded, dev_labels, eval_batch_size, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss()

    avg_loss = 0
    acc = 0
    count = 0

    preds = []
    with torch.no_grad():
        for batch in loader:
            l = batch["label"]

            s1 = batch["sentence1"]
            s2 = batch["sentence2"]

            logits = model(s1, s2).squeeze()
            loss = loss_fn(logits, l)

            avg_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            acc += (preds == l).sum()
            count += l.shape[0]

    metrics = {
        "loss": avg_loss / len(loader),
        "accuracy": (acc / count).item(),
    }

    return metrics

def train(
    model: WNLIFFNN,
    train_data_embedded: List[Dict[str, torch.Tensor]],
    train_labels: List[str],
    dev_data_embedded: List[Dict[str, torch.Tensor]],
    dev_labels: List[str],
    lr: float = 1e-3,
    batch_size: int = 32,
    eval_batch_size: int = 128,
    n_epochs: int = 10,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Runs the training loop for `n_epochs` epochs on the WNLI dataset.

    Inputs:
    - model: The WNLIFFNN model to be trained
    - train_data_embedded: List of dictionaries containing the embedded context and disambiguation for the training data
    - train_labels: List of labels for the training data
    - dev_data_embedded: List of dictionaries containing the embedded context and disambiguation for the validation data
    - dev_labels: List of labels for the validation data
    - lr: Learning rate for the optimizer
    - n_epochs: Number of epochs to train the model

    Returns:
    - train_losses: List of training losses for each epoch
    - dev_metrics: List[Dict[str, float]] of validation metrics (loss, accuracy) for each epoch
    """
    model.to(device)
    model.train()

    loader = get_wnli_dataloader(train_data_embedded, train_labels, batch_size, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr)

    train_losses = []
    dev_metrics = []

    for epoch in range(n_epochs):
        e_loss = 0

        for batch in loader:
            l = batch["label"]

            s1 = batch["sentence1"]
            s2 = batch["sentence2"]

            optimizer.zero_grad()
            logits = model(s1, s2).squeeze()
            loss = loss_fn(logits, l)

            loss.backward()
            optimizer.step()

            e_loss += loss.item()

        e_loss /= len(loader)
        train_losses.append(e_loss)

        metrics = evaluate(model, dev_data_embedded, dev_labels, eval_batch_size, device)
        dev_metrics.append(metrics)

        if verbose:
            print("Epoch: %.d, Train Loss: %.4f, Dev Loss: %.4f, Dev Accuracy: %.4f" % (epoch + 1, e_loss, metrics["loss"], metrics["accuracy"]))

    return train_losses, dev_metrics

# %% [markdown]
# # Hyperparameter Tuning

# %%
lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
n_epochs = [i * 10 for i in range(1, 11)]
batch_sizes = [2 ** i for i in range(5, 10)]
depths = [i for i in range(1, 6)]
hidden_units = [2 ** i for i in range(6, 12)]

combinations = list(product(lrs, n_epochs, batch_sizes, depths, hidden_units))
choices = len(combinations)

# %%
best_st= {"lr": 0, "epoch": 0, "batch_size": 0, "depth": 0, "width": 0, "accuracy" : 0}
for i in range(64):
    print("Trial ", i + 1)
    lr, epoch, batch_size, depth, width = combinations[np.random.choice(choices)]
    print("lr:", lr, "batch_size:", batch_size, "epoch:", epoch, "depth:", depth, "width:", width)

    model = WNLIFFNN(input_dim=768, hidden_dim=width, depth=depth)
    tl, dm = train(model, X_train_st, Y_train, X_dev_st, Y_dev, 
                           lr=lr, n_epochs=epoch, batch_size=batch_size, device='cpu', verbose=False)
    
    if dm[-1]["accuracy"] > best_st["accuracy"]:
        best_st["accuracy"] = dm[-1]["accuracy"]
        best_st["lr"] = lr
        best_st["epoch"] = epoch
        best_st["batch_size"] = batch_size
        best_st["depth"] = depth
        best_st["width"] = width
    print('Dev Accuracy:', dm[-1]["accuracy"], 'Best Accuracy:', best_st["accuracy"])
print('Best Combination:', best_st)

# %%
best_glove= {"lr": 0, "epoch": 0, "batch_size": 0, "depth": 0, "width": 0, "accuracy" : 0}
for i in range(64):
    print("Trial ", i + 1)
    lr, epoch, batch_size, depth, width = combinations[np.random.choice(choices)]
    print("lr:", lr, "batch_size:", batch_size, "epoch:", epoch, "depth:", depth, "width:", width)

    model = WNLIFFNN(input_dim=50, hidden_dim=width, depth=depth)
    tl, dm = train(model, X_train_glove, Y_train, X_dev_glove, Y_dev, 
                           lr=lr, n_epochs=epoch, batch_size=batch_size, device='cpu', verbose=False)
    
    print('Dev Accuracy:', dm[-1]["accuracy"], 'Best Accuracy:', best_glove["accuracy"])
    if dm[-1]["accuracy"] > best_glove["accuracy"]:
        best_glove["accuracy"] = dm[-1]["accuracy"]
        best_glove["lr"] = lr
        best_glove["epoch"] = epoch
        best_glove["batch_size"] = batch_size
        best_glove["depth"] = depth
        best_glove["width"] = width
print('Best Combination:', best_glove)


