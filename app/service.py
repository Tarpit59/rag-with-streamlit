import os
import logging
import re
import tempfile
import shutil
import markdown2
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec
from langchain_community.vectorstores import Pinecone
import pinecone
import cohere

def save_uploaded_files_to_temp_dir(uploaded_files):
    '''
    Saves the uploaded PDF files to a temporary directory.
    Input: List of uploaded files.
    Output: Path to the temporary directory where files are stored.
    '''
    temp_dir = tempfile.mkdtemp()
    try:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
        return temp_dir
    except Exception as e:
        logging.info(f"Error saving uploaded files: {e}")
        raise

def load_documents_from_temp_dir(temp_dir):
    '''
    Loads the PDF files from the temporary directory as documents.
    Input: Path to the temporary directory containing PDF files.
    Output: List of documents.
    '''
    try:
        print('temp_dir', temp_dir)
        file_loader = PyPDFDirectoryLoader(temp_dir)
        documents = file_loader.load()
        logging.info("All documents loaded successfully.")
        return documents
    except Exception as e:
        logging.info(f"Error loading documents: {e}")
        raise

def delete_temp_dir(temp_dir):
    '''
    Deletes the temporary directory after processing the files.
    Input: Path to the temporary directory.
    Output: None.
    '''
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        logging.info("Temporary files deleted successfully.")
    except Exception as e:
        logging.info(f"Error deleting temporary files: {e}")
        raise

def chunk_documents(docs, chunk_size=1500, chunk_overlap=150):
    '''
    Splits the documents into smaller chunks for better processing.
    Input: List of documents, chunk_size, and chunk_overlap parameters.
    Output: List of document chunks.
    '''
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(docs)
        logging.info("Documents split into chunks successfully.")
        return chunks
    except Exception as e:
        logging.error(f"Error splitting documents: {e}")
        raise

def initialize_pinecone(api_key, index_name, dimension=768, metric='cosine', cloud='aws', region='us-east-1'):
    '''
    Initializes Pinecone with the specified parameters.
    Input: API key, index name, dimension, and other configuration options.
    Output: None (Creates a Pinecone index).
    '''
    try:
        pc = pinecone.Pinecone(api_key=api_key)
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
        logging.info(f"Pinecone index '{index_name}' created successfully.")
    except Exception as e:
        logging.error(f"Error initializing Pinecone: {e}")
        raise

def create_vector_store(documents, embeddings, index_name):
    '''
    Creates a Pinecone vector store using the document embeddings.
    Input: List of document chunks, embeddings, and index name.
    Output: Pinecone vector store.
    '''
    try:
        index = Pinecone.from_documents(documents, embeddings, index_name=index_name)
        logging.info("Vector store created successfully.")
        return index
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        raise

def rerank_results_with_cohere(query, search_results, cohere_api_key, 
                               rerank_model='rerank-english-v3.0', 
                               relevance_threshold=0.001):
    '''
    Reranks search results based on relevance using Cohere's model.
    Input: Query, search results, Cohere API key, rerank model, and relevance threshold.
    Output: Reranked search results.
    '''
    try:
        co = cohere.Client(api_key=cohere_api_key)
        documents = [doc.page_content for doc in search_results]
        rerank_response = co.rerank(query=query, documents=documents, model=rerank_model)
        
        reranked_results = sorted(
            [
                (doc, result.relevance_score) 
                for doc, result in zip(search_results, rerank_response.results) 
                if result.relevance_score >= relevance_threshold  # Apply the relevance score threshold
            ],
            key=lambda x: x[1], reverse=True  # Sort by relevance score in descending order
        )
        
        logging.info("Results successfully reranked using Cohere.")
        return [result[0] for result in reranked_results]
    except Exception as e:
        logging.error(f"Error reranking results: {e}")
        raise

# Define the Document class to handle the document structure
class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content

def retrieve_answers(index, chain, query, cohere_api_key):
    '''
    Retrieves the answer using the question-answering chain and reranks results using Cohere.
    Input: Index (vector store), chain, query, and Cohere API key.
    Output: The final answer and reranked results.
    '''
    try:
        search_results = index.similarity_search(query, k=8)
        if len(search_results)==0:
            search_results = [Document(metadata='metadata', page_content='There is not relevent content')]
        reranked_results = rerank_results_with_cohere(query, search_results, cohere_api_key)
        response = chain.run(input_documents=reranked_results, question=query)
        return response, reranked_results
    except Exception as e:
        logging.error(f"Error retrieving answers: {e}")
        raise

def is_code_block(text):
    '''
    Determines if the given text is a code block.
    Input: The text to be checked.
    Output: True if the text starts with triple backticks (```), indicating a code block; False otherwise.
    '''
    return text.strip().startswith("```")

def clean_text_segment(text):
    '''
    Cleans the given text segment by converting Markdown to HTML and then removing all HTML tags.
    Input: The text segment to be cleaned.
    Output: The cleaned text segment without HTML tags.
    '''
    html_response = markdown2.markdown(text)
    clean_text = re.sub(r'<[^>]*>', '', html_response)
    return clean_text

def preserve_code_segment(code):
    '''
    Preserves the code segment while removing potential formatting characters (bold, italic, underline).
    Input: The code segment to be preserved.
    Output: The formatted code segment with triple backticks at the beginning and end.
    '''
    code = re.sub(r'(\*\*|__)', '', code)
    code = re.sub(r'(\*|_)', '', code)
    return f'```\n{code.strip()}\n```'

def process_response(response):
    '''
    Processes the given response by identifying code blocks and cleaning text segments, preserving the formatting of code blocks.
    Input: The text string representing the response.
    Output: A processed text string with code blocks preserved and text segments cleaned.
    '''
    segments = re.split(r'(```[\s\S]*?```)', response)
    processed_segments = []
    for segment in segments:
        if is_code_block(segment):
            code_content = re.sub(r'```', '', segment).strip()
            processed_segments.append(preserve_code_segment(code_content))
        else:
            processed_segments.append(clean_text_segment(segment))
    return '\n'.join(processed_segments).replace('\n\n', '\n')
