# 🔋 Battery RAG System with Memory

A sophisticated Retrieval-Augmented Generation (RAG) system for battery engineering calculations and analysis, featuring conversation memory and semantic search capabilities.

## 🌟 Features

- **📊 Data Processing**: Loads and processes battery specifications from CSV files
- **🧠 Vector Embeddings**: Uses SentenceTransformers for semantic search
- **💾 Vector Database**: ChromaDB for efficient similarity search
- **🤖 Google Gemini API**: Powered by Google's Gemini-1.5-flash model for intelligent responses
- **🗣️ Conversation Memory**: Maintains context across multiple interactions
- **⚡ Real-time Calculations**: Performs battery pack calculations (series/parallel configurations)
- **🔍 Smart Retrieval**: Finds relevant battery data based on query semantics
- **📈 Comparative Analysis**: Compares batteries across multiple parameters

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas sentence-transformers chromadb langchain langchain-google-genai python-dotenv
```

### API Setup
1. Get your Google AI Studio API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_actual_api_key_here
```

### Basic Usage
1. Set up your Google AI Studio API key in `.env` file
2. Open `rag_system.ipynb` in Jupyter
3. Run all cells in sequence to initialize the system
4. Use the direct query cell or `battery_chat()` for interaction

## 📁 Project Structure

```
battery_rag_system/
├── rag_system.ipynb           # Complete RAG system with Google Gemini API
├── battery_data_10000_rows.csv # Battery specifications dataset (10K records)
├── .env                       # Google API key configuration
└── README.md                  # Complete documentation
```

## 🔧 System Architecture

### 1. Modular Cell Structure
- Separate imports, data loading, embeddings, vector store setup
- Google Gemini API integration with fallback handling
- Individual functions for retrieval, processing, and calculations
- Clean, step-by-step implementation for easy understanding

### 2. Data Ingestion & Processing
- Loads battery specifications from CSV
- Converts to structured document format
- Handles data validation and cleaning

### 3. Vectorization & Storage
- Creates semantic embeddings using `all-MiniLM-L6-v2` model
- Stores in ChromaDB with batch processing (handles 10K+ records)
- Optimizes for fast similarity search

### 4. Google Gemini RAG Pipeline
- **Semantic Search**: ChromaDB similarity search for relevant documents
- **Context Formatting**: Structures retrieved data for optimal prompting
- **Prompt Template**: Battery engineering-specific prompt design
- **Gemini Integration**: Uses Google's Gemini-1.5-flash for intelligent responses
- **Error Handling**: Automatic fallback between Gemini models

### 5. Query Processing
- Embeds user queries for semantic matching
- Retrieves top-k most relevant battery records  
- Generates contextual responses with Google Gemini
- Provides step-by-step calculations and technical analysis

## 🤖 Google Gemini Integration

### API Configuration
```python
# Environment setup
GOOGLE_API_KEY=your_actual_api_key_here

# Model initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=google_api_key
)
```

### Usage Examples
```python
# Direct RAG query
result = rag_query("Calculate 2S3P configuration")

# Interactive chat
battery_chat()  # Starts interactive session

# Custom query
my_query = "which battery has the lowest cost per kwh"
result = rag_query(my_query)
```

## 💬 Prompt Engineering

### Main Prompt Template
```
You are a battery engineering expert. Use the provided context to answer the user's question.

Context (Retrieved Battery Data):
{context}

User Question: {query}

Instructions:
- Answer based only on the provided context
- Be technical and precise
- If calculating configurations, show step-by-step work
- For 2S3P: 2 in series (voltage adds), 3 in parallel (capacity adds)

Answer:
```

### Key Prompt Features
- **Domain Expertise**: Assumes battery engineering knowledge
- **Data Grounding**: Requires responses based on retrieved data
- **Calculation Focus**: Emphasizes step-by-step mathematical work
- **Configuration Understanding**: Handles series/parallel notation (2S3P format)
- **Technical Precision**: Ensures accurate engineering calculations

## 🧮 Calculation Capabilities

### Supported Configurations
- **Series (S)**: Voltages add, capacity remains same
- **Parallel (P)**: Capacities add, voltage remains same
- **Combined (2S3P)**: 2 in series, 3 in parallel

### Example Calculations
```
2S3P Configuration:
- Pack Voltage = Cell Voltage × 2 (series)
- Pack Capacity = Cell Capacity × 3 (parallel)
- Pack Energy = Pack Voltage × Pack Capacity
- Total Cells = 2 × 3 = 6 cells
```

## 🎯 Query Examples

### Battery Pack Calculations
```
"Calculate total energy for 4P configuration with high-density lithium batteries"
"What's the voltage and capacity of a 2S3P pack using NMC cells?"
```

### Comparative Analysis
```
"Compare the top 3 batteries by energy density"
"Which battery type is better for electric vehicles - NMC or LiFePO4?"
"Find batteries with highest capacity in the database"
```

## 🧠 Memory System Details

### Components
1. **ConversationBufferMemory**: LangChain's built-in memory for chat history
2. **Custom Query History**: Tracks interactions with metadata
3. **Context Builder**: Integrates recent conversations into prompts

### Memory Features
- Stores last 3 interactions in context
- Includes timestamps for all queries
- Maintains calculation history
- Enables follow-up questions

## 📊 Data Format

### Expected CSV Structure
```csv
Battery_ID,Type,Voltage,Capacity,Energy_Density,Weight,Chemistry,Anode,Separator
BAT-001,Cylindrical,3.7,2.5,250,45,NMC,Graphite,PE
```

### Document Format (Internal)
```
Battery ID: BAT-001
Type: Cylindrical
Voltage: 3.7 V
Capacity: 2.5 Ah
Energy Density: 250 Wh/kg
```

## 🚀 Advanced Usage

### Custom Query Function
```python
result = rag_query("Your query here")
print(result['response'])
print(f"Retrieved {len(result['retrieved_docs'])} documents")
```

### Interactive Chat
```python
battery_chat()  # Starts interactive session with Google Gemini
```

### API Key Management
```python
# Check API key status
print("Google Gemini (flash) LLM initialized successfully!")
print("LLM ready for RAG pipeline")
```

## 🔄 Key Technical Features

### Google Gemini Integration
- **Model**: Uses Google's Gemini-1.5-flash for fast, intelligent responses
- **Fallback**: Automatic fallback between Gemini models for reliability
- **API Management**: Secure API key handling through environment variables
- **Error Handling**: Comprehensive error handling for API limitations

### RAG Architecture
- **Semantic Search**: Advanced document retrieval using vector embeddings
- **Context Formation**: Intelligent context building from retrieved documents
- **Prompt Engineering**: Domain-specific prompts for battery engineering
- **Response Generation**: Technical analysis with step-by-step calculations

### Data Processing
- **CSV Integration**: Processes large battery datasets (10K+ records)
- **Document Chunking**: Converts structured data to searchable documents
- **Vector Storage**: Efficient similarity search with ChromaDB
- **Batch Processing**: Optimized for large-scale data handling

## 🛠️ Configuration Options

### Embedding Model
- Default: `all-MiniLM-L6-v2`
- Alternative: `all-MiniLM-L12-v2` (better quality, slower)

### Vector Database
- ChromaDB with in-memory storage
- Batch size: 1000 documents (adjustable)

### Google Gemini API
- Model: gemini-1.5-flash (primary)
- Fallback: gemini-1.5-pro (if available)
- Rate limiting: Automatic handling and retries

## 🆘 Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure your Google AI Studio API key is valid and properly set in `.env`
2. **ChromaDB Batch Size Error**: Reduce batch_size in the code
3. **Gemini Rate Limits**: System automatically falls back to gemini-1.5-flash model
4. **Embedding Errors**: Ensure sentence-transformers is properly installed

### Performance Tips
- Use smaller embedding models for faster processing
- Adjust top_k parameter based on your needs
- Monitor Google AI Studio usage quotas
- Check API key validity if getting authentication errors

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the example notebooks