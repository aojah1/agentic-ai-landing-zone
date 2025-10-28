import os, oci
import oracledb
import time
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from src.llm.oci_embedding_model import initialize_embedding_model

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent.parent
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


def create_ivf_index(connection):
    cur = connection.cursor()
    for ddl in [
        "BEGIN EXECUTE IMMEDIATE 'DROP INDEX vector_idx'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -1418 THEN RAISE; END IF; END;",
        "BEGIN EXECUTE IMMEDIATE 'DROP TABLE vector_table PURGE'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;"
    ]:
        cur.execute(ddl)
    cur.execute("""
        CREATE TABLE vector_table (
            id        VARCHAR2(32) PRIMARY KEY,
            embedding VECTOR(1536, FLOAT32),
            metadata  CLOB,
            text      CLOB
        )
    """)
    # IVF partitioned index
    cur.execute("""
        CREATE VECTOR INDEX vector_idx ON vector_table(embedding)
        ORGANIZATION NEIGHBOR PARTITIONS
        DISTANCE COSINE
        WITH PARAMETERS (type IVF, neighbor partitions 100)
    """)
    connection.commit()
    cur.close()


def create_hnsw_index(connection):
    cur = connection.cursor()
    for ddl in [
        "BEGIN EXECUTE IMMEDIATE 'DROP INDEX vector_idx'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -1418 THEN RAISE; END IF; END;",
        "BEGIN EXECUTE IMMEDIATE 'DROP TABLE vector_table PURGE'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;"
    ]:
        cur.execute(ddl)
    cur.execute("""
        CREATE TABLE vector_table (
            id        VARCHAR2(32) PRIMARY KEY,
            embedding VECTOR(1536, FLOAT32),
            metadata  CLOB,
            text      CLOB
        )
    """)
    # HNSW graph index
    cur.execute("""
        CREATE VECTOR INDEX vector_idx ON vector_table(embedding)
        ORGANIZATION INMEMORY NEIGHBOR GRAPH
        DISTANCE COSINE
        WITH PARAMETERS (type HNSW, neighbors 32, efconstruction 500)
    """)
    connection.commit()
    cur.close()




def main():
    """
    Tool to store docs as vectors in Oracle 23ai, with explicit vector index creation and demonstration.
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

    # RAG step5: Embed and store chunks as vectors in Oracle 23ai
    embed_model = initialize_embedding_model()

    # Establish connection to Oracle 23ai Database
    try:
        connection = oracledb.connect(
            user=DB_USER, 
            password=DB_PASSWORD, 
            dsn=DB_DSN, 
            mode=oracledb.SYSDBA  # SYSDBA mode as in your code
        )
    except oracledb.Error as e:
        raise ValueError(f"Failed to connect to Oracle DB: {e}")
    print(connection)

    # Create IVF index
    create_ivf_index(connection)

    # Initialize OracleVS (this will use the table)
    vectordb = OracleVS(
        client=connection,
        embedding_function=embed_model,
        table_name="vector_table",
        distance_strategy=DistanceStrategy.COSINE
    )

    MAX_NUMBER_DOCUMENTS_IN_VECTOR_DB = 5461
    vectordb.add_documents(documents[:MAX_NUMBER_DOCUMENTS_IN_VECTOR_DB])
    connection.commit()
    print("Chunks are stored in vector_table.")

    # Demonstrate index usage with a direct SQL query (native Oracle capability)
    demo_query = "What is Form 10 - K"  # Replace with a real query string
    
    import json

    embedding = embed_model.embed_query(demo_query)
    if len(embedding) != 1536:
        raise ValueError("Embedding length must be 1536.")

    # Bind a JSON array string to avoid giant SQL literals and type errors
    qvec = json.dumps(embedding, separators=(',', ':'))

    sql = """
    SELECT id,
        DBMS_LOB.SUBSTR(text, 300, 1) AS snippet,
        VECTOR_DISTANCE(embedding, TO_VECTOR(:qvec, 1536, FLOAT32), COSINE) AS distance
    -- or use the operator: (embedding <=> TO_VECTOR(:qvec, 1536, FLOAT32)) AS distance
    FROM vector_table
    ORDER BY distance
    FETCH FIRST 4 ROWS ONLY
    """
    cursor = connection.cursor()
    cursor.execute(sql, qvec=qvec)
    for row in cursor:
        print(f"ID: {row[0]}, Text: {row[1][:50]}..., Distance: {row[2]}")
    cursor.close()



if __name__ == "__main__":
    main()