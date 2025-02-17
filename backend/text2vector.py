from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from vector_db import neo4j_store  
PDF_STORAGE_PATH = "assets/research_paper.pdf"
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    for idx, doc in enumerate(document_chunks):
        neo4j_store.store_document(f"doc_{idx}", doc.page_content)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join(context_documents)
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

raw_docs = load_pdf_documents(PDF_STORAGE_PATH)
processed_chunks = chunk_documents(raw_docs)
index_documents(processed_chunks)

user_input = "what is the title of the research paper? and does the author acknowledges use of AI?"
relevant_docs = neo4j_store.search_similar_documents(user_input)
ai_response = generate_answer(user_input, relevant_docs)

print(ai_response)

neo4j_store.close()
