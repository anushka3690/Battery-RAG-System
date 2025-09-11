from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import traceback
import pathlib

# Load environment variables
load_dotenv()

# Set base directory - important for serverless environments
BASE_DIR = pathlib.Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
PREPROCESSED_DIR = BASE_DIR / "preprocessed"

# Initialize Flask app with explicit template and static folders for Vercel
app = Flask(__name__, 
            template_folder=str(TEMPLATES_DIR),
            static_folder=str(STATIC_DIR))

class BatteryRAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.collection = None
        self.llm = None
        self.prompt_template = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the RAG system by loading pre-processed resources"""
        try:
            print("🔄 Initializing Battery RAG System from pre-processed data...")
            
            # Check if pre-processed data exists
            if not PREPROCESSED_DIR.exists():
                print("❌ Pre-processed data not found. Please run preprocess.py first.")
                return False
            
            # Load the embedding model
            print("🧠 Loading embedding model...")
            model_path = PREPROCESSED_DIR / "embedding_model"
            if model_path.exists():
                self.embedding_model = SentenceTransformer(str(model_path))
                print("✅ Embedding model loaded from disk")
            else:
                print("⚠️ Pre-saved model not found, loading default model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Default embedding model loaded")
            
            # Connect to the vector store
            print("🗄️ Connecting to persistent vector store...")
            chroma_db_path = PREPROCESSED_DIR / "chroma_db"
            chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
            
            try:
                self.collection = chroma_client.get_collection(name="battery_rag")
                doc_count = self.collection.count()
                print(f"✅ Connected to vector store with {doc_count} documents")
            except Exception as e:
                print(f"❌ Error connecting to vector store: {e}")
                print("⚠️ The database might not exist. Please run preprocess.py first.")
                return False
            
            # Setup LLM
            print("🤖 Setting up LLM...")
            try:
                google_api_key = os.getenv('GOOGLE_API_KEY')
                if google_api_key:
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=google_api_key
                    )
                    print("✅ Google Gemini LLM initialized")
                else:
                    print("⚠️ No Google API key found. LLM responses will be limited.")
                    self.llm = None
            except Exception as e:
                print(f"⚠️ LLM initialization failed: {e}")
                self.llm = None
            
            # Setup prompt template
            self.prompt_template = PromptTemplate(
                input_variables=["query", "context"],
                template="""You are a battery engineering expert. Use the provided context to answer the user's question.

Context (Retrieved Battery Data):
{context}

User Question: {query}

Instructions:
- Answer based only on the provided context
- Be technical and precise
- If calculating configurations, show step-by-step work
- For 2S3P: 2 in series (voltage adds), 3 in parallel (capacity adds)

Answer:"""
            )
            
            self.is_initialized = True
            print("🎉 Battery RAG System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            print(traceback.format_exc())
            raise e
    
    # No need for format_battery_document as documents are pre-processed
    
    def semantic_search(self, query, top_k=3):
        """Perform semantic search to retrieve relevant documents"""
        if not self.is_initialized:
            raise ValueError("RAG system not initialized")
            
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search vector store
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        return results['documents'][0]
    
    def rag_query(self, user_query):
        """Complete RAG pipeline: Query → Search → Context → Prompt → LLM → Response"""
        try:
            if not self.is_initialized:
                return {
                    "query": user_query,
                    "response": "❌ RAG system not initialized. Please refresh the page.",
                    "retrieved_docs": [],
                    "context": ""
                }
            
            # Step 1: Semantic Search
            retrieved_docs = self.semantic_search(user_query, top_k=3)
            
            # Step 2: Format Context
            context = "\n---\n".join(retrieved_docs)
            
            # Step 3: Generate Response
            if self.llm:
                # Use LLM if available
                formatted_prompt = self.prompt_template.format(query=user_query, context=context)
                llm_response = self.llm.invoke(formatted_prompt)
                
                if hasattr(llm_response, 'content'):
                    response = llm_response.content
                else:
                    response = str(llm_response)
            else:
                # Fallback response without LLM
                response = f"""Based on the retrieved battery data context:

{context[:500]}...

The system found {len(retrieved_docs)} relevant battery entries for your query: "{user_query}"

Note: For enhanced AI responses, please configure your Google API key in the .env file."""
            
            return {
                "query": user_query,
                "retrieved_docs": retrieved_docs,
                "response": response,
                "context": context
            }
            
        except Exception as e:
            return {
                "query": user_query,
                "response": f"❌ Error processing query: {str(e)}",
                "retrieved_docs": [],
                "context": ""
            }

# Initialize RAG system
rag_system = BatteryRAGSystem()

@app.route('/')
def index():
    """Main page"""
    # Try to initialize if not already done
    if not rag_system.is_initialized:
        init_rag_if_preprocessed()
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize_system():
    """Initialize the RAG system from pre-processed data"""
    try:
        if not os.path.exists('preprocessed'):
            return jsonify({
                "status": "error",
                "message": "❌ Pre-processed data not found. Please run preprocess.py first."
            })
        
        success = rag_system.initialize()
        if success:
            return jsonify({
                "status": "success",
                "message": "🎉 Battery RAG System initialized successfully from pre-processed data!"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "❌ Could not initialize the RAG system. Please run preprocess.py first."
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"❌ Error initializing system: {str(e)}"
        })

@app.route('/query', methods=['POST'])
def query():
    """Process user query through RAG system"""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({
                "status": "error",
                "message": "Please enter a valid query"
            })
        
        # Check if system is initialized
        if not rag_system.is_initialized:
            return jsonify({
                "status": "error",
                "message": "❌ System not initialized. Please click the initialization banner first."
            })
        
        # Process through RAG system
        result = rag_system.rag_query(user_query)
        
        return jsonify({
            "status": "success",
            "result": result
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"❌ Error processing query: {str(e)}"
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "initialized": rag_system.is_initialized,
        "llm_available": rag_system.llm is not None
    })

# Initialize RAG system if preprocessed data exists
def init_rag_if_preprocessed():
    """Try to initialize the RAG system if preprocessed data exists"""
    if PREPROCESSED_DIR.exists():
        try:
            if not rag_system.is_initialized:
                print("🔄 Attempting to initialize RAG system...")
                rag_system.initialize()
                return True
        except Exception as e:
            print(f"⚠️ Could not auto-initialize RAG system: {e}")
            print("ℹ️ You can still initialize it manually via the UI.")
    return False

# Create a function that initializes the app - this is used by the WSGI entry point
def create_app():
    """Create and configure the Flask app for production/Vercel deployment"""
    # Set up logging for Vercel
    if os.environ.get('VERCEL_ENV') == 'production':
        # Vercel-specific configuration
        app.config['DEBUG'] = False
        app.config['TEMPLATES_AUTO_RELOAD'] = False
    else:
        app.config['DEBUG'] = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        
    # Try to initialize the RAG system
    with app.app_context():
        init_success = init_rag_if_preprocessed()
        if init_success:
            print("✅ RAG system automatically initialized on startup")
            
    return app

if __name__ == '__main__':
    print("🔋 Starting Battery RAG System Web App...")
    
    # Check if pre-processed data exists
    if PREPROCESSED_DIR.exists():
        print("✅ Pre-processed data found. The system will initialize automatically.")
    else:
        print("⚠️ Pre-processed data not found.")
        print("🔄 Please run 'python preprocess.py' before using the system for best performance.")
    
    # Initialize and run the app for local development
    application = create_app()
    application.run(debug=True, host='0.0.0.0', port=5000)
