
import os
import json
from pydantic import BaseModel, Field
from langchain_docling import DoclingLoader 
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma 
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap

# --- CONFIGURATION ---
PDF_PATH = r''#enter the path of your PDF here
CHROMA_DIR = "./chroma_db_tata" # Using a unique directory name
COLLECTION_NAME = " " # Enter your PDF path.
EMBEDDING_MODEL = 'mxbai-embed-large'
LLM_MODEL = 'llama3'

# --- 1. UTILITY FUNCTIONS ---

def format_source_documents(docs: list[Document]) -> list[str]:
    """Formats the retrieved documents into a simple list of content strings with metadata."""
    # This formats the metadata for easy verification
    return [
        f"SOURCE: {d.metadata.get('source', 'N/A')}\nCHUNK ID: {d.metadata.get('chunk_id', 'N/A')}\nCONTENT:\n---{d.page_content}---" 
        for d in docs
    ]

class ESGMetric(BaseModel):
    """Structured output for ESG metric extraction."""
    metric_name: str = Field(description="Waste intensity per rupee of turnover (MT/₹)'")
    value: float = Field(description="The numerical value extracted from the context. Must be a float, no commas.")
    unit: str = Field(description="(MT/₹)")
    year: str = Field(description="The reporting period")
    source_chunk_id: int = Field(description="The chunk_id where the data was found.")

# --- 2. DATA INGESTION (Run only if the DB doesn't exist) ---

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    print("🔄 INGESTION MODE: Starting PDF processing...")
    
    # a. Load & Structure Data using Docling
    loader = DoclingLoader(PDF_PATH)
    docs_from_loader = loader.load()
    text = "\n\n".join([d.page_content for d in docs_from_loader])

    # b. Semantic Chunking
    text_splitter = SemanticChunker(embeddings=embeddings)
    chunk_strings = text_splitter.split_text(text)

    # c. Convert to Documents with Metadata
    all_splits = []
    source_file = os.path.basename(PDF_PATH)
    for i, chunk_text in enumerate(chunk_strings):
        all_splits.append(
            Document(
                page_content=chunk_text,
                metadata={"source": source_file, "chunk_id": i + 1}
            )
        )
    print(f"📄 Total semantic chunks created: {len(all_splits)}")

    # d. Create and Persist the Vector Store
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        persist_directory=CHROMA_DIR,
        embedding=embeddings,
        collection_name=COLLECTION_NAME
    )
    print(f"✅ ChromaDB created and saved to {CHROMA_DIR}")

else:
    print(f"♻️ LOADING MODE: Loading existing ChromaDB from {CHROMA_DIR}...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    print("✅ Database loaded successfully.")


# --- 3. RETRIEVER & GENERATION SETUP ---

# a. Initialize Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} 
)
print("✅ Retriever initialized and configured to fetch 5 chunks.")

# b. Initialize LLM and Parser
parser = JsonOutputParser(pydantic_object=ESGMetric)
# IMPORTANT: format="json" forces Ollama to output valid JSON for the parser
llm = ChatOllama(model=LLM_MODEL, temperature=0.1, format="json") 

# c. Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert ESG Data Extraction Agent. Your primary goal is to extract data as it is found in the provided CONTEXT. "
               "If the value is in thousands, extract the floating point number without rounding. "
               "The output **MUST BE** a single JSON object that strictly follows this schema. "
               "If the data is truly missing, set the 'value' field to **0.0** and the 'unit' field to **'NA'**. "
               "Do not output any text before or after the JSON object.\n\n{format_instructions}"),
    ("human", "CONTEXT:\n---{context}---\n\nQUESTION: {question}")
])

# --- 4. THE RAG CHAIN (Traceability Included) ---

# Define the core extraction pipeline (runs prompt -> llm -> parser)
rag_extraction_pipeline = (
    {
        "context": retriever, 
        "question": RunnablePassthrough(), 
        "format_instructions": lambda x: parser.get_format_instructions()
    }
    | prompt | llm | parser
)

# Define the retrieval pipeline that gets the chunks and formats them for display
retrieval_pipeline = (
    retriever | format_source_documents
)

# Combine both pipelines into a single chain that runs parallel and returns both results
rag_chain_with_context = RunnableMap(
    extracted_data=rag_extraction_pipeline,
    source_chunks=retrieval_pipeline
)

# --- 5. EXECUTION ---

# The query to execute
query_to_extract = "What is the company's Total waste generated (in metric tonnes) in FY 24 ?"

print("\n--- Running Extraction Chain ---")

# The entire RAG process runs here
result = rag_chain_with_context.invoke(query_to_extract)

print("\n" + "="*70)
print(f"🎯 Query: {query_to_extract}")
print("="*70)

# Output the structured result
print("\n--- FINAL STRUCTURED EXTRACTION ---")
if result['extracted_data'] is None:
    print("No data extracted.")
print(json.dumps(result['extracted_data'], indent=4))

# Output the source chunks
print("\n--- SOURCE CHUNKS FOR VERIFICATION ---")
for chunk in result['source_chunks']:
    print(chunk)
    print("-" * 70)