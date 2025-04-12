import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_rag_model(data, model_name='facebook/rag-sequence-nq', epochs=3, batch_size=8, learning_rate=1e-5):
    """
    Train a RAG model on the provided data.
    
    Args:
        data (list): List of training examples.
        model_name (str): Name of the pre-trained RAG model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for training.
        
    Returns:
        model: Trained RAG model.
    """
    tokenizer = RagTokenizer.from_pretrained(model_name)
    retriever = RagRetriever.from_pretrained(model_name, index_name="exact", passages_path="data/passage_data")
    model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_data, val_data = train_test_split(data, test_size=0.1)
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        val_loss = 0
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
                val_loss += outputs.loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_data)}")
    
    return model

def evaluate_rag_model(model, data, batch_size=8):
    """
    Evaluate the RAG model on the provided data.
    
    Args:
        model: Trained RAG model.
        data (list): List of evaluation examples.
        batch_size (int): Batch size for evaluation.
        
    Returns:
        float: Accuracy of the model on the evaluation data.
    """
    tokenizer = RagTokenizer.from_pretrained(model.config.name_or_path)
    
    model.eval()
    predictions = []
    references = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs['input_ids'])
            predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
            references.extend(batch)
    
    accuracy = accuracy_score(references, predictions)
    return accuracy

def generate_race_summaries(model, race_data, batch_size=8):
    """
    Generate race summaries using the trained RAG model.
    
    Args:
        model: Trained RAG model.
        race_data (list): List of race data examples.
        batch_size (int): Batch size for generation.
        
    Returns:
        list: List of generated race summaries.
    """
    tokenizer = RagTokenizer.from_pretrained(model.config.name_or_path)
    
    model.eval()
    summaries = []
    for i in range(0, len(race_data), batch_size):
        batch = race_data[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs['input_ids'])
            summaries.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    
    return summaries
