import boto3
import json
import streamlit as st

bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

prompt = """
<s>[INST]You are a summarization system that can provide summaries with associated confidence 
scores. In clear and concise language, provide three short summaries of the following essay, 
along with their confidence scores. You will respond with a pretty response with Summary 
and Confidence. Do not provide explanations.[/INST]

# Essay: 
{Essay}
"""

modelId = "mistral.mistral-large-2402-v1:0"

accept = "application/json"
contentType = "application/json"


def streamlit_ui():
    st.set_page_config("Mistral<>Bedrock")
    st.header("Text Summarization using Mistral's Large and AWS Bedrock")

    user_question = st.text_area("Provide me with a text to summarize.")

    if st.button("Summarize") or user_question:
        if not user_question:
            st.error("Please provide a text to summarize.")
            return
        with st.spinner("Processing..."):
            filled_prompt = prompt.format(Essay=user_question)
            body = json.dumps({
                "prompt": filled_prompt,
                "max_tokens": 512,
                "top_p": 0.8,
                "temperature": 0.5,
            })
            response = bedrock.invoke_model(
                body=body,
                modelId=modelId,
                accept=accept,
                contentType=contentType
            )
            response_json = json.loads(response["body"].read())
            text = response_json['outputs'][0]['text']
            st.write(text)
            st.success('Done!')


if __name__ == "__main__":
    streamlit_ui()
