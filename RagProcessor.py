import os
import time
import vector_store_helper
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


class RagProcessor:
    '''
    A class that will handle all the interactions with a RAG
    '''
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        self.is_dev_Call = os.getenv('DEV_CALL')

        self.chat_gpt_models = ['gpt-3.5-turbo', 'gpt-4',
                                'gpt-4-turbo-preview', 'gpt-4o']
        self.default_open_ai_chat_model = self.chat_gpt_models[3]

        self.anthropic_models = ['claude-instant-1.2', 'claude-2.0',
                                 'claude-2.1', 'claude-3-haiku-20240307',
                                 'claude-3-sonnet-20240229',
                                 'claude-3-opus-20240229']
        self.default_anthropic_model = self.anthropic_models[5]

        self.prompt_template = """
        Answer the question based only on the following context:
        {context}

        Question: {question}
        """

    async def fetch_answer_from_rag_using_collections(
            self, question: str, prompt: str, use_model: str,
            company_name: str, embedding_model_name: str, collection_name: str,
            use_local_qdrant_vectorstore: bool, vector_store_name: str):
        '''
        A method that fetches an answer based on a question from a RAG

        Parameters
        ==========
        question: string
            The question to be answered
        prompt: string, NOT REQUIRED
            A prompt to be used along side the question.
        use_model: string, NOT REQUIRED
            The llm to use
        company_name: string
            The company name the index belongs to

        Returns
        =======
        qnc: JSON
            A json data structure with keys 'answer' and 'context'
            containing the answer of the question asked and the context from
            which it was generated from
        '''
        print(f'\n\n\n\n=============> Incoming question: {question}')
        # step 1 - load the vector store
        try:
            start_time = time.time()
            vector_store = await vector_store_helper.initialize_vector_store(
                                    embedding_model_name, collection_name,
                                    use_local_qdrant_vectorstore,
                                    vector_store=vector_store_name)
            retriever = vector_store.as_retriever()
            time_taken=round(time.time()-start_time, 2)
            print('\nretriever: {retriever}\nloaded from vector store in: '
                  + f'{time_taken} ...')
            print(f'\ntype of the retriever is: {type(retriever)} ...')
        except Exception as e2:
            error_message = f"An error occurred while loading the retriever from the vector store.\nError: {e2}"
            print(error_message)
            return {"error_message": error_message,
                    "status_code": 500,
                    "status": 'failed'
                    }

        # step 2 - create a prompt template, llm and a chain
        try:
            start_time = time.time()
            # setup prompt
            if prompt is None or prompt == "":
                prompt = ChatPromptTemplate.from_template(
                    self.prompt_template)       # type: ignore
            else:
                prompt = ChatPromptTemplate.from_template(
                    prompt)     # type: ignore
            print(f'\n\ncurrently used prompt: {prompt}')

            # setup llm
            if use_model is None or use_model == "" or use_model == "claude":
                llm_model = ChatAnthropic(model=self.default_anthropic_model,
                                      anthropic_api_key=self.ANTHROPIC_API_KEY)
            elif use_model == "openai":
                llm_model = ChatOpenAI(model=self.default_open_ai_chat_model,
                                       api_key=self.OPENAI_API_KEY)
            else:
                error_message = f'\nThe provided LLM: {use_model} can not be found'
                print(error_message)
                return {"error_message": error_message,
                        "status_code": 400,
                        "status": 'failed'
                        }
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm_model
                | StrOutputParser()
            ).with_config({"run_name": 'fetch_answer_from_rag_using_collections_EP_slim_api' + "\'s run"})
            time_taken = round(time.time()-start_time, 2)
            print(f'\n\nchain: {chain}\ncreated in {time_taken}')
        except Exception as e3:
            error_message = f'\nAn error occurred while creating a prompt and a chain.\nError: {e3}'
            print(error_message)
            return {"error_message": error_message,
                    "status_code": 500,
                    "status": 'failed'
                    }

        # step 3 - generate answer
        try:
            start_time = time.time()
            print('\n\ngenerating answers ...')
            langsmith_tracer = LangChainTracer(project_name=company_name)
            answer = chain.invoke(question,
                                  config={"callbacks": [langsmith_tracer],
                                          "tags": [company_name],
                                          "metadata": {
                                              "vector_store": vector_store,
                                              "retriever": retriever,
                                              "dev call": self.is_dev_Call,
                                              "used llm": llm_model
                                          }
                                          }
                                  )
            context_retrieved_from_index = retriever.get_relevant_documents(
                question)
            time_taken = round(time.time()-start_time, 2)
            print(f'answer and context generation completed in: {time_taken}')
            return {
                "answer": answer,
                "context": context_retrieved_from_index,
                "status_code": 200,
                "status": 'success'
            }
        except Exception as e4:
            error_message = f'\nAn error occurred while generating answer and context.\nError: {e4}'
            print(error_message)
            return {"error_message": error_message,
                    "status_code": 500,
                    "status": 'failed'
                    }
