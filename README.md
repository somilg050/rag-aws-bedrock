# RAG implementation using AWS Bedrock and LlamaIndex/LangChain

### Basic RAG application to query our PDFs and fetch relevant answers. You need to have a basic understanding of Python for this tutorial.

Frameworks/Technologies we will use:
* AWS Bedrock (a fully managed service that makes leading foundation models available through a Unified API)
* LlamaIndex/LangChain (a data framework for LLMs that helps developers work with data)
* Streamlit (a free, open-source framework that allows users to create and share web apps from Python scripts)


### The processing of this application involves two components:
1. Prepare the documents (VectorStore)

![image](https://github.com/somilg050/rag-aws-bedrock/assets/31178867/3c30ce1a-b40f-4ada-a62b-2457a7d713da)

2. Retrieve the relevant documents and frame the response using LLM.

![image](https://github.com/somilg050/rag-aws-bedrock/assets/31178867/cbbe43a8-0b2c-4b35-8e5d-c760a3abb02d)


---

* **Retrieval of Relevant Documents:** The model searches a large corpus of documents to find the most relevant to the input query. 

* **Response Generation Using LLM (Large Language Model):** Once the relevant documents are retrieved, a large language model uses the information from these documents to generate a coherent and contextually appropriate response. 

---
