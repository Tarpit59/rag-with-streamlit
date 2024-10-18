from service import save_uploaded_files_to_temp_dir, load_documents_from_temp_dir, delete_temp_dir, \
                    chunk_documents,initialize_pinecone, create_vector_store, Document, retrieve_answers, \
                    process_response
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

INDEX_NAME = "langcha"

# Streamlit app
st.title("Interactive QA Bot with PDF")

st.subheader("Enter API Keys")
google_api_key = st.text_input("Google API Key", type="password")
groq_api_key = st.text_input("Groq API Key", type="password")
cohere_api_key = st.text_input("Cohere API Key", type="password")
pinecone_api_key = st.text_input("Pinecone API Key", type="password")

# Show message if not all API keys are provided
if not (google_api_key and groq_api_key and cohere_api_key and pinecone_api_key):
    st.warning("Please enter all required API keys to enable saving.")

# Save API keys button (disabled if not all keys are provided)
save_button_disabled = not (google_api_key and groq_api_key and cohere_api_key and pinecone_api_key)
if st.button("Save API Keys", disabled=save_button_disabled):
    st.session_state.GOOGLE_API_KEY = google_api_key
    st.session_state.GROQ_API_KEY = groq_api_key
    st.session_state.COHERE_API_KEY = cohere_api_key

    os.environ['PINECONE_API_KEY'] = pinecone_api_key
    st.success("API Keys have been saved successfully!")

GOOGLE_API_KEY = st.session_state.get("GOOGLE_API_KEY")
GROQ_API_KEY = st.session_state.get("GROQ_API_KEY")
COHERE_API_KEY = st.session_state.get("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if GOOGLE_API_KEY and GROQ_API_KEY and COHERE_API_KEY and PINECONE_API_KEY:
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

    # Initialize Pinecone
    if "pinecone_initialized" not in st.session_state:
        initialize_pinecone(api_key=PINECONE_API_KEY, index_name="langcha")
        st.session_state.pinecone_initialized = True

    # Initialize QA chain
    if "chain_initialized" not in st.session_state:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
        chain = load_qa_chain(llm, chain_type="stuff")
        st.session_state.chain_initialized = True
        st.session_state.chain = chain

    # Upload PDFs
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files]

        if new_files:
            temp_dir = save_uploaded_files_to_temp_dir(uploaded_files)
            documents = load_documents_from_temp_dir(temp_dir)

            chunks = chunk_documents(documents)
            if "document_chunks" not in st.session_state:
                st.session_state.document_chunks = chunks
            else:
                st.session_state.document_chunks.extend(chunks)

            if "vector_store" in st.session_state and st.session_state.vector_store:
                st.session_state.vector_store.add_documents(chunks)
            else:
                vector_store = create_vector_store(st.session_state.document_chunks, embeddings, "langcha")
                st.session_state.vector_store = vector_store

            delete_temp_dir(temp_dir)

            st.session_state.uploaded_files.extend([f.name for f in new_files])

    # Enter user query
    with st.form(key='query_form'):
        query = st.text_input("Enter your query", key="query")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            st.session_state.submitted = True

            if "vector_store" not in st.session_state or not st.session_state.vector_store:
                st.error("Please upload PDF documents before submitting a query.")
            elif not query:
                st.error("Please enter a query before submitting.")
            else:
                vector_store = st.session_state.vector_store
                chain = st.session_state.chain
                answer, reranked_results = retrieve_answers(vector_store, chain, query, cohere_api_key=COHERE_API_KEY)
                processed_response = process_response(answer)

                st.subheader("Answer:")
                st.write(processed_response)

                processed_documents = []

                for doc in reranked_results:
                    content = doc.page_content
                    processed_content = process_response(content)
                    processed_documents.append(Document(metadata=doc.metadata, page_content=processed_content))

                st.subheader("Relevant Document Sections:")
                for i, doc in enumerate(processed_documents):
                    st.write('Document: ', i + 1)
                    st.write(doc.page_content)
else:
    st.warning("Please enter API Keys to proceed.")
