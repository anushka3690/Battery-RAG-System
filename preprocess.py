"""
Preprocessing script for Battery RAG System
This script processes the battery data, creates embeddings, and saves them to disk
Run this script before starting the Flask app to have everything pre-loaded
"""
import pandas as pd
import numpy as np
import os
import chromadb
import pickle
import pathlib
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set base directory for consistent path handling
BASE_DIR = pathlib.Path(__file__).parent
PREPROCESSED_DIR = BASE_DIR / "preprocessed"

def format_battery_document(row):
    """Convert battery row into structured text document"""
    return f"""Battery ID: {row.iloc[0]}
Type: {row.iloc[1] if len(row) > 1 else 'Unknown'}
Voltage: {row.iloc[2] if len(row) > 2 else 'Unknown'} V
Capacity: {row.iloc[3] if len(row) > 3 else 'Unknown'} Ah
Energy Density: {row.iloc[4] if len(row) > 4 else 'Unknown'} Wh/kg"""

def preprocess_data():
    """Preprocess the data and save the resources to disk"""
    print("🔄 Running offline preprocessing for Battery RAG System...")
    
    # Create a directory for preprocessed resources if it doesn't exist
    PREPROCESSED_DIR.mkdir(exist_ok=True)
    
    # Load and process data
    print("📊 Loading battery data...")
    data_file = BASE_DIR / 'battery_data_10000_rows.csv'
    df = pd.read_csv(data_file)
    df = df.apply(pd.to_numeric, errors='ignore').dropna().reset_index(drop=True)
    print(f"✅ Loaded {len(df)} battery records")
    
    # Create documents
    print("📄 Creating structured documents...")
    documents = [format_battery_document(row) for _, row in df.iterrows()]
    print(f"✅ Created {len(documents)} structured documents")
    
    # Save documents
    documents_path = PREPROCESSED_DIR / 'documents.pkl'
    with open(documents_path, 'wb') as f:
        pickle.dump(documents, f)
    print("✅ Saved documents to disk")
    
    # Initialize embedding model
    print("🧠 Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedding model loaded")
    
    # Create embeddings
    print("🧠 Creating embeddings...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    print(f"✅ Created embeddings with shape: {embeddings.shape}")
    
    # Save embeddings
    embeddings_path = PREPROCESSED_DIR / 'embeddings.npy'
    np.save(embeddings_path, embeddings)
    print("✅ Saved embeddings to disk")
    
    # Save embedding model for consistent encoding later
    model_path = PREPROCESSED_DIR / 'embedding_model'
    embedding_model.save(str(model_path))
    print("✅ Saved embedding model to disk")
    
    # Setup vector store
    print("🗄️ Setting up persistent vector store...")
    chroma_db_path = PREPROCESSED_DIR / 'chroma_db'
    chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
    
    # Clean up existing collection
    try:
        chroma_client.delete_collection(name="battery_rag")
    except:
        pass
        
    collection = chroma_client.create_collection(name="battery_rag")
    
    # Store documents in batches
    print("💾 Storing documents in vector store...")
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            embeddings=embeddings[i:end_idx].tolist(),
            ids=[str(j) for j in range(i, end_idx)]
        )
        print(f"   Batch {i//batch_size + 1}: {end_idx - i} documents")
    
    print(f"✅ Stored {collection.count()} documents in persistent vector store")
    print("🎉 Preprocessing complete! You can now start the Flask app with a lighter footprint.")

if __name__ == "__main__":
    preprocess_data()
