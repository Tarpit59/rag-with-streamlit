
# RAG-Based Interactive QA Bot with PDF

This project implements a Retrieval-Augmented Generation (RAG) bot that allows users to upload PDF documents, ask questions, and receive answers based on the content of the uploaded files. The app is containerized using Docker for easy deployment and is designed for scalability and ease of use.

## Features
- Upload multiple PDF documents
- Interactive QA bot powered by the LangChain and Pinecone
- Integration with Google and Cohere API for embeddings and reranking
- Dockerized for ease of deployment
- Scalable to handle large documents and multiple queries
- User-friendly frontend built using Streamlit

## Technologies Used
- **Python 3.9**
- **Streamlit** for frontend interface
- **LangChain** for document processing and question answering
- **Pinecone** for vector storage
- **Google Generative AI Embeddings** for vectorization
- **Cohere API** for reranking answers
- **Docker** for containerization

## Installation

#### 1. Clone the repository:

```bash
git clone https://github.com/Tarpit59/rag-with-streamlit.git
cd rag-with-streamlit
```
#### 2. Build the Docker container:

```bash
docker build -t qa-bot .
```

#### 3. Run the container:

```bash
docker run -p 8501:8501 qa-bot
```

#### 4. Access the app in your browser at http://localhost:8501.

## Usage

### Upload documents
1. After launching the app, you will be prompted to enter your API keys (Google, Groq, Cohere, and Pinecone).
2. Once the keys are saved, upload one or more PDF files through the "Upload PDF documents" button.
### Ask Questions
1. Enter your query in the text input field after uploading documents.
2. Click the "Submit" button to ask your question.
### View Answers and Relevant Sections
1. After submitting a query, the bot will provide an answer based on the document content.
2. Below the answer, relevant sections from the documents will also be displayed.

## Example Interactions
Here’s an example interaction with the bot:

- **Uploaded PDF:** A research paper on artificial intelligence.
- **Query:** "What are the latest advancements in AI?"
- **Bot's Response:** "The latest advancements in AI include - improvements in language models, neural architecture search, and reinforcement learning."

#### Relevant Document Sections:
- Document 1: Page 5 - "Advancements in language models have enabled..."

## Colab Notebook
[Documentation](https://github.com/Tarpit59/rag-with-streamlit/blob/3d482273fefefbb7bedf9aac83a763bb828429e4/notebook/README.md)

For convenience, a fully functional Colab notebook is available for running the model without the need for local setup. The notebook covers the entire pipeline:

- Loading documents
- Creating vector stores
- Asking questions
- Generating answers

You can access the Colab notebook [here.](https://github.com/Tarpit59/rag-with-streamlit/blob/3d482273fefefbb7bedf9aac83a763bb828429e4/notebook/Colab_notebook.ipynb)

## Deployment
### Docker Deployment

**1.** Build the Docker image:
```bash
docker build -t qa-bot .
```

**2.** Run the container:
```bash
docker run -p 8501:8501 qa-bot
```

**3.** Access the app in your browser at http://localhost:8501.

## Challenges and Solutions
1. **Handling large documents:** The system efficiently chunks documents using LangChain’s RecursiveCharacterTextSplitter to handle large documents and optimize memory usage.

2. **Multiple queries performance:** Pinecone’s vector store, combined with Google embeddings and Cohere’s reranking model, ensures fast and accurate results even with multiple queries.

3. **Environment management:** Using Docker simplifies deployment and avoids dependency issues across different platforms.

## Acknowledgements

 - [LangChain RAG](https://python.langchain.com/v0.2/docs/tutorials/rag/)
 - [Cohere ReRank with LangChain](https://docs.cohere.com/docs/rerank-on-langchain#:~:text=for%20more%20information.-,Cohere%20ReRank%20with%20LangChain,retrievers%2C%20embeddings%2C%20and%20RAG.)
- [Pinecone guides](https://docs.pinecone.io/guides/get-started/quickstart)
- [ChatGroq](https://python.langchain.com/docs/integrations/chat/groq/)
## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tarpit-patel)
