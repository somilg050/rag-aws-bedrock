import os
import boto3
import streamlit as st

# In this example, we'll use the AWS Titan Embeddings model to generate embeddings.
# You can use any model that generates embeddings.
from langchain_community.embeddings.bedrock import BedrockEmbeddings

# Load the Titan Embeddings using Bedrock client.
bedrock = boto3.client(service_name='bedrock-runtime')
titan_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                     client=bedrock)

# Vector Store for Vector Embeddings
from langchain_community.vectorstores.faiss import FAISS

# Load RetrievalQA from langchain as it provides a simple interface to interact with the LLM.
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

# Imports for Data Ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

# Import Bedrock for LLM
from langchain_community.llms.bedrock import Bedrock

# Load the PDFs from the directory
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=250)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Store for Vector Embeddings
def setup_vector_store(documents):
    # Create a vector store using FAISS from the documents and the embeddings
    vector_store = FAISS.from_documents(
        documents,
        titan_embeddings,
    )
    # Save the vector store locally
    vector_store.save_local("faiss_index")

# Load the LLM from the Bedrock
def load_llm():
    llm = Bedrock(model_id="mistral.mistral-large-2402-v1:0", client=bedrock, model_kwargs={"max_tokens": 512})
    return llm

# Create a prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If the answer is not within the context knowledge, kindly state that you do not know, rather than attempting to fabricate a response.
2. If you find the answer, please craft a detailed and concise response to the question at the end. Aim for a summary of max 250 words, ensuring that your explanation is thorough.

{context}

Question: {question}
Helpful Answer:"""

# Now we use langchain PromptTemplate to create the prompt template for our LLM
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create a RetrievalQA chain and invoke the LLM
def get_response(llm, vector_store, query):
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    response = retrieval_qa.invoke(query)
    return response['result']

def streamlit_ui():
    st.set_page_config("My Gita RAG")
    st.header("RAG implementation using AWS Bedrock and Langchain")

    user_question = st.text_input("Ask me anything from My Gita e.g. What is the meaning of life?")

    with st.sidebar:
        st.title("Update Or Create Vector Embeddings")

        if st.button("Update Vector Store"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                setup_vector_store(docs)
                st.success("Done")

    if st.button("Generate Response") or user_question:
        # first check if the vector store exists
        if not os.path.exists("faiss_index"):
            st.error("Please create the vector store first from the sidebar.")
            return
        if not user_question:
            st.error("Please enter a question.")
            return
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", embeddings=titan_embeddings,
                                           allow_dangerous_deserialization=True)
            llm = load_llm()
            st.write(get_response(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    streamlit_ui()