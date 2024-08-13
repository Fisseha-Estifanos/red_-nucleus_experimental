import os
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


async def initialize_vector_store(embedding_model_name: str,
                                  collection_name: str,
                                  use_local_qdrant_vectorstore: bool,
                                  vector_store: str):
    try:
        # embedding model setup
        if embedding_model_name == "text-embedding-3-large":
            embedding_model = OpenAIEmbeddings(model=embedding_model_name,
                                               dimensions=1024)
        elif embedding_model_name == "mxbai-embed-large":
            embedding_model = OllamaEmbeddings(model=embedding_model_name)

        # vector store setup
        if vector_store == "qdrant":
            if use_local_qdrant_vectorstore:
                qdrant_url = os.environ["QDRANT_LOCAL_URL"]
                qdrant_client = QdrantClient(url=qdrant_url, timeout=120)
            else:
                qdrant_url = os.environ["QDRANT_CLOUD_URL"]
                qdrant_client = QdrantClient(
                    url=qdrant_url, timeout=120,
                    api_key=os.environ['QDRANT_API_KEY'])
            qdrant_vector_store = Qdrant(
                client=qdrant_client, collection_name=collection_name,
                embeddings=embedding_model)
            print("---> Qdrant vector store to be initialized : "
                  + f"{qdrant_vector_store}\n--->"
                  + f" Type : {type(qdrant_vector_store)}")
            print(f"\t---> embedding_model: {embedding_model}")
            print(f"\t---> vector store url: {qdrant_url}")
            print(f"\t---> collection name: {collection_name}")
            return qdrant_vector_store
        elif vector_store == "pg_vector":
            local_connection = os.environ["LOCAL_DATABASE_URL"]
            pg_vector_vectorstore = PGVector(
                embeddings=embedding_model,
                collection_name=collection_name,
                connection=local_connection,
                use_jsonb=True,
            )
            print("---> PGvector vector store to be initialized : "
                  + f"{pg_vector_vectorstore}\n--->"
                  + f" Type : {type(pg_vector_vectorstore)}")
            print(f"\t---> embedding_model: {embedding_model}")
            print(f"\t---> Connection url: {local_connection}")
            print(f"\t---> collection name: {collection_name}")
            return pg_vector_vectorstore
    except Exception as ex:
        print('Exception occurred while trying to initialize the vector store.'
              + f'\nError: {ex}')
