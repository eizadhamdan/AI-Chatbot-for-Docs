from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")

    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key
    )

    return embeddings
