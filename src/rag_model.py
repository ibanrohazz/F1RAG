"""
Formula 1 RAG (Retrieval Augmented Generation) Model
This script trains and generates summaries using a RAG model with Formula 1 race data.
"""
import os
import json
import argparse
import logging
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing transformers components with error handling
try:
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM, 
        RagTokenizer, RagRetriever, RagSequenceForGeneration,
        DPRContextEncoder, DPRContextEncoderTokenizer,
    )
    logger.info("Successfully imported transformers library")
except ImportError as e:
    logger.error(f"Error importing transformers: {e}")
    logger.error("Please install transformers library using: pip install transformers")
    raise

class F1RaceDataset(Dataset):
    """Dataset for F1 race data"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_passage_data(data, output_dir='data/passage_data'):
    """
    Create passage data for RAG model retrieval
    
    Args:
        data (list): List of text examples
        output_dir (str): Directory to save passage data
        
    Returns:
        dict: Dictionary with passages and their IDs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure data is a list of strings
    text_examples = []
    for item in data:
        if isinstance(item, str):
            text_examples.append(item)
        else:
            # Skip non-string items
            logger.warning(f"Skipping non-string item during passage data creation: {type(item)}")
    
    if not text_examples:
        logger.warning("No valid text examples for passage creation. Creating a dummy passage.")
        text_examples = ["Formula 1 racing data placeholder"]
    
    logger.info(f"Creating passages from {len(text_examples)} text examples")
    
    # Create simple passage collection from the data
    passages = []
    for i, text in enumerate(text_examples):
        passages.append({
            'id': str(i),
            'text': text,
            'title': f'Formula 1 Data {i}'
        })
    
    # Save passages to files needed by RAG
    with open(os.path.join(output_dir, 'passages.json'), 'w') as f:
        json.dump(passages, f)
    
    # Skip complex embedding creation to avoid NumPy version conflicts
    logger.info("Skipping passage embeddings creation to avoid NumPy compatibility issues")
    return {'passages': passages}

def train_rag_model(data, model_name='facebook/rag-sequence-nq', epochs=3, batch_size=4, learning_rate=1e-5, output_dir='models'):
    """
    Train a RAG model on the provided data.
    
    Args:
        data (list): List of training examples.
        model_name (str): Name of the pre-trained RAG model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for training.
        output_dir (str): Directory to save the trained model.
        
    Returns:
        model: Trained RAG model.
    """
    logger.info(f"Initializing RAG model with {model_name}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create passage data for retrieval
    logger.info("Creating passage data for retrieval...")
    passage_data = create_passage_data(data)
    
    # Ensure all data items are strings
    string_data = []
    for item in data:
        if isinstance(item, str):
            string_data.append(item)
        elif isinstance(item, dict):
            # Convert dict to string representation
            string_data.append(json.dumps(item))
        else:
            # Try to convert to string
            try:
                string_data.append(str(item))
            except:
                logger.warning(f"Skipping item that can't be converted to string: {type(item)}")
    
    if not string_data:
        logger.error("No valid string data for training")
        raise ValueError("No valid string data for training")
        
    logger.info(f"Prepared {len(string_data)} string examples for training")

    # Try to use seq2seq model for training
    try:
        # Initialize tokenizer and model
        logger.info("Loading tokenizer and model...")
        # Simple approach using seq2seq model
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        logger.info("Splitting data into training and validation sets")
        train_data, val_data = train_test_split(string_data, test_size=0.1, random_state=42)
        train_dataset = F1RaceDataset(train_data)
        val_dataset = F1RaceDataset(val_data)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size
        )
        logger.info(f"Starting training for {epochs} epochs")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model.to(device)
        best_val_loss = float('inf')
        try:
            for epoch in range(epochs):
                model.train()
                total_train_loss = 0
                logger.info(f"Starting epoch {epoch+1}/{epochs}")
                for batch_idx, batch in enumerate(train_dataloader):
                    if not all(isinstance(item, str) for item in batch):
                        logger.warning(f"Batch contains non-string items. Converting to strings.")
                        batch = [str(item) for item in batch]
                    inputs = tokenizer(batch, padding=True, truncation=True, 
                                      max_length=512, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    labels = inputs["input_ids"].clone()
                    outputs = model(input_ids=inputs['input_ids'], 
                                    attention_mask=inputs.get('attention_mask'), 
                                    labels=labels)
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                    if (batch_idx + 1) % 5 == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
                avg_train_loss = total_train_loss / len(train_dataloader)
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        if not all(isinstance(item, str) for item in batch):
                            batch = [str(item) for item in batch]
                        inputs = tokenizer(batch, padding=True, truncation=True, 
                                          max_length=512, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        labels = inputs["input_ids"].clone()
                        outputs = model(input_ids=inputs['input_ids'], 
                                       attention_mask=inputs.get('attention_mask'), 
                                       labels=labels)
                        total_val_loss += outputs.loss.item()
                avg_val_loss = total_val_loss / len(val_dataloader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    logger.info(f"New best validation loss: {best_val_loss:.4f}, saving model")
                    model_path = os.path.join(output_dir, f"model_epoch_{epoch+1}")
                    model.save_pretrained(model_path)
                    tokenizer.save_pretrained(model_path)
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user (Ctrl+C). Saving current model...")
            interrupted_model_path = os.path.join(output_dir, "model_interrupted")
            model.save_pretrained(interrupted_model_path)
            tokenizer.save_pretrained(interrupted_model_path)
            logger.info(f"Model saved to {interrupted_model_path} after interruption.")
            return model, tokenizer
        final_model_path = os.path.join(output_dir, "model_final")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Training complete. Final model saved to {final_model_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        logger.error("Failed to train the model.")
        raise

def generate_race_summaries(model, tokenizer, race_data, batch_size=4, max_length=150):
    """
    Generate race summaries using the trained model.
    
    Args:
        model: Trained model.
        tokenizer: Corresponding tokenizer.
        race_data (list): List of race data examples.
        batch_size (int): Batch size for generation.
        max_length (int): Maximum length of generated summaries.
        
    Returns:
        list: List of generated race summaries.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Ensure all items are strings
    string_data = []
    for item in race_data:
        if isinstance(item, str):
            string_data.append(item)
        else:
            try:
                string_data.append(str(item))
            except:
                logger.warning(f"Skipping item that can't be converted to string during generation")
    
    if not string_data:
        logger.error("No valid string data for generation")
        return []
        
    logger.info(f"Generating summaries for {len(string_data)} examples")
    
    summaries = []
    
    for i in range(0, len(string_data), batch_size):
        batch = string_data[i:i+batch_size]
        
        inputs = tokenizer(batch, padding=True, truncation=True, 
                           return_tensors="pt", max_length=512).to(device)
        
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        decoded_summaries = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        summaries.extend(decoded_summaries)
        
        logger.info(f"Generated {len(decoded_summaries)} summaries, total: {len(summaries)}/{len(string_data)}")
    
    return summaries

def load_data(data_file='data/processed/race_data.json'):
    """Load and prepare data for the model"""
    logger.info(f"Loading data from {data_file}")
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Handle case where data is a dict of race info
            examples = []
            # Check if we have lap_times
            if 'lap_times' in data:
                for driver in data['lap_times'].keys():
                    example = f"Driver: {driver}\n"
                    example += f"Race: {data.get('race_name', 'Unknown')}\n"
                    example += f"Circuit: {data.get('circuit_name', 'Unknown')}\n"
                    
                    lap_times = data['lap_times'][driver][:5]  # First 5 lap times
                    if lap_times:
                        lap_time_str = ', '.join(str(lt) for lt in lap_times)
                        example += f"Lap Times: {lap_time_str}...\n"
                    
                    examples.append(example)
            else:
                # Handle case where data dict doesn't have lap_times
                examples = [f"F1 Data: {json.dumps(data)}"]
        elif isinstance(data, list):
            # Convert each item in the list to a string if it's not already
            examples = []
            for item in data:
                if isinstance(item, str):
                    examples.append(item)
                elif isinstance(item, dict):
                    # Format dictionary data as a structured string
                    text = ""
                    for key, value in item.items():
                        if key == "lap_times" and isinstance(value, list) and len(value) > 5:
                            # Limit lap times to first 5
                            lap_times_str = ', '.join(str(lt) for lt in value[:5])
                            text += f"{key}: {lap_times_str}...\n"
                        else:
                            text += f"{key}: {value}\n"
                    examples.append(text)
                else:
                    # For any other type, convert to string
                    examples.append(str(item))
        else:
            logger.error(f"Unexpected data format: {type(data)}")
            examples = [str(data)]
        
        logger.info(f"Loaded {len(examples)} examples")
        return examples
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Create a minimal example if data loading fails
        return ["Formula 1 racing data example."]

def create_sample_data(output_file='data/processed/sample_data.json'):
    """Create a minimal sample dataset if no real data exists"""
    sample_data = {
        "race_name": "Sample Grand Prix",
        "circuit_name": "Sample Circuit",
        "lap_times": {
            "Driver 1": ["1:33.421", "1:32.876", "1:32.543", "1:32.111", "1:31.998"],
            "Driver 2": ["1:33.876", "1:32.912", "1:32.654", "1:32.432", "1:32.123"],
            "Driver 3": ["1:34.123", "1:33.432", "1:33.021", "1:32.765", "1:32.432"]
        }
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
        
    logger.info(f"Created sample data at {output_file}")
    return sample_data

def main():
    """Main function for training and generating with the model"""
    parser = argparse.ArgumentParser(description="Train model or generate race summaries")
    parser.add_argument("--generate", action="store_true", help="Generate race summaries")
    parser.add_argument("--data-file", type=str, default="data/processed/race_data.json", 
                        help="Path to processed race data")
    parser.add_argument("--model-dir", type=str, default="models/model_final", 
                        help="Path to trained model directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, 
                        help="Learning rate for training")
    parser.add_argument("--output-dir", type=str, default="models", 
                        help="Directory to save the model")
    parser.add_argument("--sample", action="store_true", 
                        help="Create and use sample data if no real data exists")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug information about the data")
    
    args = parser.parse_args()
    
    # Check for data file or create sample
    if not os.path.exists(args.data_file):
        logger.warning(f"Data file not found: {args.data_file}")
        if args.sample:
            create_sample_data()
            args.data_file = "data/processed/sample_data.json"
        else:
            logger.error("No data file and --sample not specified. Exiting.")
            return
    
    # Load data
    examples = load_data(args.data_file)
    
    if not examples:
        logger.error("No examples loaded, exiting.")
        return
    
    # Debug info
    if args.debug:
        logger.info(f"Data types: {[type(ex) for ex in examples[:5]]}")
        logger.info(f"Sample data item: {examples[0][:200]}...")
    
    if args.generate:
        try:
            # Load the trained model for generation
            logger.info(f"Loading model from {args.model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
            
            # Generate summaries
            logger.info("Generating race summaries")
            summaries = generate_race_summaries(
                model, tokenizer, examples, batch_size=args.batch_size
            )
            
            # Save summaries
            os.makedirs("data/output", exist_ok=True)
            output_file = "data/output/race_summaries.json"
            with open(output_file, 'w') as f:
                json.dump(summaries, f, indent=2)
            
            logger.info(f"Generated {len(summaries)} summaries, saved to {output_file}")
            
            # Display a few examples
            logger.info("Example summaries:")
            for i, summary in enumerate(summaries[:3]):
                logger.info(f"Summary {i+1}: {summary[:100]}...")
                
        except Exception as e:
            logger.error(f"Error in generating summaries: {e}")
    else:
        # Train the model
        logger.info("Training model...")
        try:
            model, tokenizer = train_rag_model(
                examples,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir
            )
            logger.info("Model training completed")
            
            # Save a small config file with model information
            config = {
                "model_type": "seq2seq",
                "base_model": "facebook/bart-large",
                "training_examples": len(examples),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate
            }
            
            with open(os.path.join(args.output_dir, "model_info.json"), 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error in model training: {e}")

if __name__ == "__main__":
    main()
