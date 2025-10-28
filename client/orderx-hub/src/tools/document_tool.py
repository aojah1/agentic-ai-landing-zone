""" 
upload or reference unstructured documents for semantic search and retrieval upon which to ground an answer or response
Build this using Langchain or LlamaIndex
Tools :
1) Document Loader: Store Vector Use Case: Text Splitter --> Document building --> Embedding --> Save in 23.AI
2) Document Retriever : Retrieval of data : Prompt text splitter --> Prompt Embedding --> Semantic Search --> Rerank

## Understanding RAG Workflow with StuffDocumentsChain

**RAG (Retrieval-Augmented Generation)** is a powerful workflow that combines:
-  **Retriever** (e.g., FAISS, ChromaDB): Fetches relevant documents from a large knowledge base.
-  **LLM** (e.g., Cohere, OpenAI): Answers questions using those documents.
-  **Document Chains** like `StuffDocumentsChain`: Combine documents into a prompt.

###  Components:
- **Document Loader**: Loads raw documents from files (e.g., `.txt`, `.pdf`, etc.).
- **Text Splitter**: Breaks long documents into chunks to fit within LLM context window.
- **Vector Store (FAISS)**: Stores vector embeddings of text chunks and retrieves the most similar ones.
- **Cohere Embeddings**: Convert text into vectors using Cohere's model (`embed-english-v3.0`).
- **Chat Model (Cohere)**: Generates answers based on provided documents and questions.
- **StuffDocumentsChain**: Concatenates all retrieved docs and "stuffs" them into a single prompt.
- **LangChain Expression Language (LCEL)**: Defines modular, reusable pipelines using operators like `|`.

This notebook demonstrates a complete RAG system using a finance dataset and `StuffDocumentsChain`.

"""

import oracledb
import oci, os
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from src.llm.oci_embedding_model import initialize_embedding_model
from io import BytesIO
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from oci.addons.adk import Toolkit, tool
from pathlib import Path
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent
print(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / "config/.env")  # expects OCI_ vars in .env

# Set up the OCI GenAI Agents endpoint configuration
BUCKET_REGION = os.getenv("BUCKET_REGION")
print("BUCKET_REGION: " + BUCKET_REGION)
BUCKET_NAME = os.getenv("BUCKET_NAME")
print("BUCKET_NAME: " + BUCKET_NAME)
BUCKET_PDF_NAME = os.getenv("BUCKET_PDF_NAME")
print("BUCKET_PDF_NAME: " + BUCKET_PDF_NAME)
OBJECT_STORAGE_NAMESPACE_NAME = os.getenv("OBJECT_STORAGE_NAMESPACE_NAME")
print("OBJECT_STORAGE_NAMESPACE_NAME: " + OBJECT_STORAGE_NAMESPACE_NAME)
LOCAL_PDF_PATH = os.getenv("LOCAL_PDF_PATH")
print("LOCAL_PDF_PATH: " + LOCAL_PDF_PATH)
USE_LOCAL_PDF = os.getenv("USE_LOCAL_PDF")
print("USE_LOCAL_PDF: " + USE_LOCAL_PDF)

# Set up Oracle Database connection (replace with your actual connection details)
# These should be loaded from environment variables or config for security
DB_USER = os.getenv("DB_USER")  # e.g., your Oracle DB username
DB_PASSWORD = os.getenv("DB_PASSWORD")  # e.g., your Oracle DB password
DB_DSN = os.getenv("DB_DSN")  # e.g., "your-host:1521/your-service-name" or ADB connection string



### Tool to store docs as vectors in Oracle 23ai
@tool
def store_documents():
    """
    Tool to store docs as vectors in Oracle 23ai
    """

    # RAG Step1 : Load the PDF document and create pdf reader object

    if USE_LOCAL_PDF=="true":
        loader = PyPDFLoader(LOCAL_PDF_PATH)
        pages = loader.load()
    else:
        # OCI Object Storage configuration
        config = oci.config.from_file()
        object_storage = oci.object_storage.ObjectStorageClient(config)

        # Bucket and object details
        # Download the PDF from OCI Object Storage
        response = object_storage.get_object(OBJECT_STORAGE_NAMESPACE_NAME, BUCKET_NAME, BUCKET_PDF_NAME)
        pdf_content = response.data.content

        # Load the PDF into PyPDFLoader
        pdf_bytes = BytesIO(pdf_content)

        ## Save pdf_bytes to a temporary file
        # pdf_bytes is a BytesIO object
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes.getvalue())  # Extract raw bytes
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        ## clean up temporary file
        os.remove(tmp_file_path)


    ## RAG step2 : Combined raw text from all pages
    raw_text = ''

    for i, doc in enumerate(pages):
        text = doc.page_content
        if text:
            raw_text += text

    print("You have transformed the PDF document to text format")

    ## RAG step3 : Chunk the text into smaller chunks
    ### Text splitter

    text_splitter = RecursiveCharacterTextSplitter(
        # chunk_size should enable as many characters as can fit in max number of tokens allowed for embedding model\
        #  i.e. 512 characters. Source: https://docs.oracle.com/en-us/iaas/Content/generative-ai/use-playground-embed.htm
        chunk_size=2000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(raw_text)
    print(len(chunks))

    ## RAG step4 : Text chunk into a document structure

    documents = []
    for i in range(len(chunks)):
        doc = Document(page_content=chunks[i])
        # doc = texts[i]
        documents.append(doc)

    
    # RAG step5 : using an embedding model embed the chunks as vectors into oracle database 23ai

    ## Initialize a vector database (replaced Chroma with Oracle 23ai Vector Search)
    embed_model = initialize_embedding_model()

     # Establish connection to Oracle 23ai Database
    # Use a context manager for safe connection handling
    try:
        connection = oracledb.connect(
            user=DB_USER, 
            password=DB_PASSWORD, 
            dsn=DB_DSN, 
            mode=oracledb.SYSDBA  # SYSDBA mode as in your code
        )
    except oracledb.Error as e:
        raise ValueError(f"Failed to connect to Oracle DB: {e}")


    # Initialize Oracle Vector Store
    # Assumes a table named 'vector_table' exists or will be created with vector columns.
    # You may need to create the table beforehand with appropriate schema (e.g., ID, VECTOR(embedding_dim, FLOAT32), METADATA, CONTENT)
    vectordb = OracleVS(
        client=connection,
        embedding_function=embed_model,
        table_name="vector_table",  # Table name for storing vectors
        distance_strategy=DistanceStrategy.COSINE,  # Or DOT_PRODUCT, EUCLIDEAN, etc.
    )
    

    MAX_NUMBER_DOCUMENTS_IN_VECTOR_DB = 5461

    # Add documents to Oracle Vector Store (this will embed and insert into the DB table)
    vectordb.add_documents(documents[:MAX_NUMBER_DOCUMENTS_IN_VECTOR_DB])

    print("Chunks are stored in the vector_table")

    # Clean up: Close connection
    connection.close()

# #%% md
### Tool Creation to retrive docs from Oracle 23ai

@tool
def retrieve_documents(query: str) -> dict:
    """
    Tool to retrieve vectorized docs from Oracle 23ai.
    """
    
    # Establish connection to Oracle 23ai Database
    # Use a context manager for safe connection handling
    try:
        connection = oracledb.connect(
            user=DB_USER, 
            password=DB_PASSWORD, 
            dsn=DB_DSN, 
            mode=oracledb.SYSDBA  # SYSDBA mode as in your code
        )
    except oracledb.Error as e:
        raise ValueError(f"Failed to connect to Oracle DB: {e}")

    # Initialize embedding model (assuming this function is defined elsewhere)
    embed_model = initialize_embedding_model()

    # Initialize Oracle Vector Store
    # Removed 'query_type' as it's not a valid parameter
    # Assumes 'vector_table' exists; create it if needed (see tips below)
    vectordb = OracleVS(
        client=connection,
        embedding_function=embed_model,
        table_name="vector_table",  # Table name for storing vectors
        distance_strategy=DistanceStrategy.COSINE  # Valid param: COSINE, DOT_PRODUCT, etc.
        # Optional: Add id_column="id", vector_column="vector" if custom schema
    )

    # Create a retriever with semantic/similarity search
    retriever = vectordb.as_retriever(
        search_type="similarity",  # Enables semantic search (instead of query_type)
        search_kwargs={
            "k": 4,  # Number of top results to retrieve
            # Optional: "score_threshold": 0.8 for filtering low-relevance results
        }
    )

    # Retrieve relevant documents
    content_docs = retriever.invoke(query)
    
    # Extract and concatenate content (as in your commented-out code)
    raw_text = ''
    for doc in content_docs:
        raw_text += doc.page_content + '\n'  # Add newline for readability

    # Clean up: Close connection
    connection.close()

    # Return as dict (matching your original intent)
    return {"content": raw_text}

def test_retrieve_documents():
    str1 = "List different events from year 2020 to 2023"
    retrieve_documents(str1)


@tool
def rag_execute() -> str:
    ## Define a prompt template

    PRODUCT_BOT_PROMPT = """
        You are a smart assistant.
        Your response must only be in English.
        Ensure your answers are relevant to the query with reference to provided context and not outside the context.
        Your responses should be elaborate and up to the mark referring to the context only.
        Do not include the keyword context in the final answer

        CONTEXT:
        {context}

        QUESTION: {question}

        YOUR ANSWER:
    """

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_PROMPT)

    ## Chain the inputs and outputs

    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from src.llm.oci_genai import initialize_llm

    llm = initialize_llm()

    # Define the full processing pipeline
    # 1. Takes query input
    # 2. Retrieves relevant context from vector DB
    # 3. Fills in the prompt template
    # 4. Sends it to the LLM
    # 5. Parses the output string

    chain = (
            {"context": lambda x: retrieve_documents(x)["content"] if x.strip() else "", "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    ## Process the query

    """ Example Questions to try
    Who is Mr. Raza
    What is Form 10 - K
    Explain acquisitions
    List different events from year 2020 to 2023
    """

    query = "What is Form 10 - K"

    result = chain.invoke(query)

    return result

def test_vs_initialization():
    
    # Sample connection (replace with your creds)
    connection = oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN, mode=oracledb.SYSDBA)

    embed_model = initialize_embedding_model()
    
    try:
        vectordb = OracleVS(
            client=connection,
            embedding_function=embed_model,
            table_name="vector_table"
        )
        print("OracleVS initialized successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    test_vs_initialization()
    store_documents()
    result = test_retrieve_documents()
    print(result)
    result1 = rag_execute()
   ## Print response
    print("Response:", result1)
    
