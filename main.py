import os
import uvicorn
from fastapi import FastAPI, HTTPException
from data_models import transfer_data_models
import RagProcessor
from utils import get_variables


app = FastAPI(swagger_ui_parameters={"syntaxHighlight": True})


@app.get('/')
async def root():
    return {
        "status": "online",
        "version": 0.01
    }


# region Rag
@app.post('/memory_test')
async def memory_test(
            rag_data: transfer_data_models.FetchAnswerUsingCollections):
    answer_fetcher = RagProcessor.RagProcessor()
    status = await answer_fetcher.fetch_answer_from_rag_using_collections(
        rag_data.question, rag_data.prompt, rag_data.use_model,
        rag_data.company_name, rag_data.embedding_model_name,
        rag_data.collection_name, rag_data.use_local_qdrant_vectorstore,
        rag_data.vector_store_name)
    print(f'\nstatus: {status}')
    if status['status'] == 'failed':
        raise HTTPException(status_code=status['status_code'],
                            detail=status['error_message'])
    elif status['status'] == 'success':
        return {
            "answer": status['answer'],
            "context": status['context']
        }
# endregion Rag

if __name__ == '__main__':
    get_variables()
    print(f'main server::- starting: {app}')
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app="main:app",
                host="0.0.0.0",
                port=port)
    print(f'main server::- over and out: {app}')
