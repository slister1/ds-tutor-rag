import os
from dotenv import load_dotenv, find_dotenv

import pinecone

from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings


def create_index():
    loader = DirectoryLoader(
        '../data/',
        glob='**/*.pdf',
        show_progress=True,
        loader_cls=PyPDFLoader
    )
    docs = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(
    text_splitter = CharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    docs_split = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model=os.environ["EMBEDDINGS_MODEL_NAME"],
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )

    Pinecone.from_documents(
        docs_split,
        embeddings,
        index_name=os.environ["PINECONE_INDEX"]
    )


def main():
    _ = load_dotenv(find_dotenv())  # read local .env file

    create_index()


if __name__ == "__main__":
    main()
