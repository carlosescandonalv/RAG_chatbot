from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import os


def get_embeddings(OPENAI_API_KEY):
    embeddings = OpenAIEmbeddings( model= "text-embedding-ada-002", openai_api_key = OPENAI_API_KEY)
    return embeddings



def get_prompt(question:str, OPENAI_API_KEY):
    db = Chroma(persist_directory="db_carlos_final", embedding_function=get_embeddings(OPENAI_API_KEY))
    retriever = db.as_retriever(seach_type="similarity")
    relevant_chunks = retriever.invoke(f"{question}")
    PROMPT_TEMPLATE = """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer
        the question. If you don't know the answer, say that you
        don't know. DON'T MAKE UP ANYTHING.

        {context}

        ---
        Answer the question based on the above context: {question}
        """
    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])

    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, 
                                    question=f"{question}")
    return prompt,relevant_chunks


def get_response(question:str,OPENAI_API_KEY):
    prompt,relevant_chunks = get_prompt(question,OPENAI_API_KEY)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    return llm.invoke(prompt).content,relevant_chunks


def process_relevant_chunks(relevant_chunks):
    data = []
    
    for doc in relevant_chunks:
        metadata = doc.metadata
        data.append({
            "document_name": metadata.get("document_name", "").split("/")[-1],  # Extract filename
            "page": metadata.get("page", ""),  # Page number
            "content": doc.page_content.strip()  # Clean text content
        })
    
    return pd.DataFrame(data)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        document_name = chunk.metadata.get("document_name")
        page = chunk.metadata.get("page")
        current_page_id = f"{document_name}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks: list, vectorstore_path: str, OPENAI_API_KEY: str):
    # Load the existing database.
    db = Chroma(
        persist_directory= vectorstore_path, embedding_function=get_embeddings(OPENAI_API_KEY)
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")




def upload_file_in_db(file_path:str, OPENAI_API_KEY:str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    document_name = os.path.basename(file_path)
    for page in pages:
        page.metadata["document_name"] = document_name
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, 
                                                chunk_overlap=200, 
                                                length_function=len, separators = ["\n\n", "\n", ""])
    chunks = text_splitter.split_documents(pages)   
    chunks = calculate_chunk_ids(chunks)
    add_to_chroma(chunks, "app/db_carlos_final", OPENAI_API_KEY)