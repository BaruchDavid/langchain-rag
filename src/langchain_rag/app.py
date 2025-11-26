#upload your pdf file and ask question
import tempfile
from pathlib import Path

import streamlit
import weaviate
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.prompts import PromptTemplate

from langchain import hub

load_dotenv()

streamlit.title("LangChain OpenAI RAG")

llm = AzureChatOpenAI(model="gpt-4o")
rag_template = PromptTemplate.from_template("Please answer this question '{question}' and only use the following context '{context}'")
client = weaviate.connect_to_local()
collection_name = "document_chunks_202506"
if client.collections.exists(collection_name):
    client.collections.delete(collection_name)

file = streamlit.file_uploader("Upload your document")
prompt = streamlit.text_input("What do you want to know?")
if file and prompt:
    # ------- Indexing -------
    # Parse PDF file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        file_path = Path(temp_file.name)

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Create chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # Create embeddings and store in Weaviate
        vector_store = WeaviateVectorStore(
            client=client,
            index_name=collection_name,
            text_key="text",
            embedding=AzureOpenAIEmbeddings(model="text-embedding-3-large"),
        )

        vector_store.add_documents(all_splits)

        # ------- Retrieval -------
        # Search for relevant chunks
        retrieved_docs = vector_store.similarity_search(prompt)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Query the LLM with the prompt
        langhchain_prompt = rag_template.invoke({"question": prompt, "context": docs_content})
        
        answer = llm.invoke(langhchain_prompt)

        streamlit.write(answer.content)