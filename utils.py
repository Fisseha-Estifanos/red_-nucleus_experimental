import os
from dotenv import load_dotenv
load_dotenv()


def get_variables():
    print('loading variables ...')
    os.environ['LANGCHAIN_API_KEY'] = os.environ["LANGCHAIN_API_KEY"]
    os.environ['LANGCHAIN_TRACING_V2'] = os.environ['LANGCHAIN_TRACING_V2']
    os.environ['LANGCHAIN_ENDPOINT'] = os.environ['LANGCHAIN_ENDPOINT']
    print('finished loading variables ...')
