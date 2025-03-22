from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma
import ollama
import os
import shutil


class DBUtils:
    CHROMA_PATH = "chroma"
    _db_instance = None

    def __init__(self):
        self._db_instance = None;
    
    def dummy(self):
        return "dummy"

    def get_chroma_path(self):
        return self.CHROMA_PATH

    def get_chroma(self, chroma_path=CHROMA_PATH, model_type="m3e"):
        if self._db_instance is None:
            self._db_instance = Chroma(
                persist_directory=chroma_path, 
                embedding_function=self.get_embedding_function(model_type)
            )
        return self._db_instance

    def clear_database(self, chroma_path=CHROMA_PATH):
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
        self._db_instance = None

    def get_embedding_function(self, model_type="m3e"):
        if model_type == "m3e":
            embeddings = HuggingFaceEmbeddings(
                model_name="moka-ai/m3e-base",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif model_type == "ollama":
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
        else:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        return embeddings
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )

    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-large",
    #     # With the `text-embedding-3` class
    #     # of models, you can specify the size
    #     # of the embeddings you want returned.
    #     # dimensions=1024
    # )


# if __name__ == "__main__":
#     # ollama pull
#     response = ollama.embeddings(model='nomic-embed-text', prompt='The sky is blue because of rayleigh scattering')
#     print(response)
    # ollama_emb = get_embedding_function()
    # r1 = ollama_emb.embed_documents(
    #     [
    #         "Alpha is the first letter of Greek alphabet",
    #         "Beta is the second letter of Greek alphabet",
    #     ]
    # )
    # r2 = ollama_emb.embed_query(
    #     "What is the second letter of Greek alphabet"
    # )
    # print(r1)
    # print("------------------------------------")
    # print(r2)
