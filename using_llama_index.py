import os.path
import streamlit as st

import boto3
bedrock = boto3.client(service_name='bedrock-runtime')

# In this example, we'll use the AWS Titan Embeddings model to generate
# embeddings. You can use any model that generates embeddings.
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings, get_response_synthesizer

# Load the Titan Embeddings using Bedrock client.
Settings.embed_model = BedrockEmbedding(model="amazon.titan-embed-text-v1",
                                        client=bedrock)

# Vector Store for Vector Embeddings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)

# Load the Bedrock from llama_index
from llama_index.llms.bedrock import Bedrock

PERSIST_DIR = "./storage"


# Load the PDFs from the directory and create a vector index
def create_index():
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index


# Load the LLM from the Bedrock
def load_llm():
    llm = Bedrock(model="amazon.titan-text-express-v1", client=bedrock, max_tokens=512)
    return llm


# Query the engine
def get_response(index, llm, user_question):
    # Create a query engine with the retriever and llm
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(user_question)
    return response


def streamlit_ui():
    st.set_page_config("My Gita RAG")
    st.header("RAG implementation using AWS Bedrock and Llama Index")

    user_question = st.text_input("Ask me anything from My Gita e.g. "
                                  "What is the meaning of life?")

    with st.sidebar:
        st.title("Update Or Create Vector Embeddings")

        if st.button("Update Vector Store"):
            with st.spinner("Processing..."):
                index = create_index()
                index.storage_context.persist(persist_dir=PERSIST_DIR)
                st.success("Done")

    if st.button("Generate Response"):
        if not os.path.exists(PERSIST_DIR):
            st.error("Please create the vector store first from the sidebar.")
            return
        if not user_question:
            st.error("Please enter a question.")
            return
        with st.spinner("Processing..."):
            llm = load_llm()
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)

            st.success(get_response(index, llm, user_question))


if __name__ == "__main__":
    streamlit_ui()
