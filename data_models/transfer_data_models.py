from pydantic import BaseModel


class FetchAnswerUsingCollections(BaseModel):
    question: str
    prompt: str
    use_model: str
    company_name: str
    embedding_model_name: str
    collection_name: str
    use_local_qdrant_vectorstore: bool
    vector_store_name: str
