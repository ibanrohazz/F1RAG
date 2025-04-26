# 🏎️ Formula 1 RAG: Conversational QA (FAISS Version)

# ---
# 📂 Step 1: Load the Formula 1 Dataset

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import faiss
from tqdm import tqdm

# (Assume CSVs are available in the same folder or adjust path)
races = pd.read_csv('races.csv')
drivers = pd.read_csv('drivers.csv')
results = pd.read_csv('results.csv')
constructors = pd.read_csv('constructors.csv')

print("Loaded:", len(races), "races,", len(drivers), "drivers,", len(results), "results.")

# ---
# 🔧 Step 2: Build the F1 Knowledge Base

# Merge to create rich descriptions per race result
f1_data = results.merge(races, on='raceId')
f1_data = f1_data.merge(drivers, on='driverId')
f1_data = f1_data.merge(constructors, on='constructorId')

# Filter only winning results (positionOrder == 1)
winners = f1_data[f1_data['positionOrder'] == 1]

# Create a text field for each winning race
winners['fact'] = winners.apply(lambda row: f"In {row['year']}, {row['forename']} {row['surname']} won the {row['name']} driving for {row['name_y']}.", axis=1)

# Final F1 Knowledge Base
f1_facts = winners[['fact']].reset_index(drop=True)
print("Sample Fact:", f1_facts.iloc[0]['fact'])

# ---
# 🧐 Step 3: Embed the Knowledge Base

# Load sentence transformer model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = AutoModel.from_pretrained(model_name)

# Function to compute BERT embeddings
def compute_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, dim=1)
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

# Embed all facts
fact_embeddings = compute_embeddings(f1_facts['fact'].tolist())
print("Embedded", fact_embeddings.shape[0], "facts.")

# ---
# 🔎 Step 4: Define the FAISS Retriever

class F1FAISSRetriever:
    def __init__(self, facts, embeddings):
        self.facts = facts
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.numpy())

    def retrieve(self, query, k=3):
        # Embed the query
        inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            output = bert(**inputs)
            query_emb = output.last_hidden_state[:, 0]
            query_emb = F.normalize(query_emb, dim=1)

        # Search in FAISS
        D, I = self.index.search(query_emb.cpu().numpy(), k)

        # Return top-k facts
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx != -1:
                results.append((self.facts[idx], score))
        return results

retriever = F1FAISSRetriever(f1_facts['fact'].tolist(), fact_embeddings)

# ---
# 💬 Step 5: Conversational Chatbot

class F1Chatbot:
    def __init__(self, retriever):
        self.retriever = retriever
        self.chat_history = []

    def chat(self, query, top_k=3):
        self.chat_history.append({"user": query})
        results = self.retriever.retrieve(query, k=top_k)

        answer = "\n".join([f"- {fact}" for fact, score in results])
        self.chat_history.append({"bot": answer})

        print(f"\nUser: {query}")
        print(f"Bot:\n{answer}\n")

    def show_history(self):
        for turn in self.chat_history:
            for speaker, text in turn.items():
                print(f"{speaker.capitalize()}: {text}\n")

# Initialize Chatbot
f1_chatbot = F1Chatbot(retriever)

# ---
# 💬 Example Chat

f1_chatbot.chat("Who won the Monaco Grand Prix in 2019?")
f1_chatbot.chat("How about Silverstone in 2014?")
f1_chatbot.chat("Who was champion in 2008?")

# ---
# 📊 (Optional) Save FAISS index for future use

# faiss.write_index(retriever.index, 'f1_facts.index')
# To load: retriever.index = faiss.read_index('f1_facts.index')
