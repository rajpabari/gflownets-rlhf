# %% [markdown]
# <a href="https://colab.research.google.com/github/rajpabari/gflownets-rlhf/blob/main/train_reward_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%

# %%
import numpy as np  # linear algebra
import pandas as pd
import numpy as np
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score
from statistics import mean
from sklearn.metrics import hamming_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score, confusion_matrix
import statistics
from sklearn.metrics import recall_score

from wordcloud import WordCloud
from collections import Counter
from tqdm import tqdm

from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# %%
import zipfile

with zipfile.ZipFile("data/train_reward.csv.zip", "r") as zip_ref:
    zip_ref.extractall("./")


# %%
train_data = pd.read_csv("./train_reward.csv")
train_data["comment_text"] = train_data["comment1"] + ":" + train_data["comment2"]
train_data.drop(["Unnamed: 0", "comment1", "comment2"], inplace=True, axis=1)
train_data.head()

# %%
from torch import cuda

device = torch.device("cuda" if cuda.is_available() else "cpu")

print(f"Current device: {device}")

# %%
train_data["comment_text"] = train_data["comment_text"].str.lower()
train_data["comment_text"] = (
    train_data["comment_text"]
    .str.replace("\xa0", " ", regex=False)
    .str.split()
    .str.join(" ")
)
train_data["score"] = train_data["score"].apply(lambda x: [x])

# %%
MAX_LEN = 512
TRAIN_BATCH_SIZE = 20
EPOCHS = 10
LEARNING_RATE = 1e-05
NUM_WORKERS = 2


# %%
class LabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len: int, eval_mode: bool = False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.comment_text
        self.eval_mode = eval_mode
        if self.eval_mode is False:
            self.targets = self.data.score
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        output = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

        if self.eval_mode is False:
            output["targets"] = torch.tensor(
                self.targets.iloc[index], dtype=torch.float
            )

        return output


# %%
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", truncation=True, do_lower_case=True
)
training_set = LabelDataset(train_data, tokenizer, MAX_LEN)

# %%
train_params = {
    "batch_size": TRAIN_BATCH_SIZE,
    "shuffle": True,
    "num_workers": NUM_WORKERS,
}
training_loader = DataLoader(training_set, **train_params)


# %%
class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        linear = self.classifier(pooler)
        output = torch.sigmoid(linear)
        return output


# %%
model = DistilBERTClass()
#parallelize the model over multiple gpus
model = torch.nn.DataParallel(model)
model.to(device)


# %%
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


# %%
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# %%
def train(epoch):
    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")

        loss.backward()
        optimizer.step()


# %%
import gc

gc.collect()
print("Beginning training")
for epoch in range(EPOCHS):
    torch.save(model.state_dict(), "./saved-models/reward_model_" + str(epoch) + ".pt")
    print("Epoch", epoch)
    train(epoch)

torch.save(model.state_dict(), "./saved-models/reward_model_final.pt")
