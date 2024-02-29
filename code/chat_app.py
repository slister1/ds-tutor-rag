import os
from dotenv import load_dotenv, find_dotenv

import streamlit as st
import openai
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone

# read local .env file
_ = load_dotenv(find_dotenv())

# Check if environment variables are present. If not, throw an error
if os.getenv('PINECONE_API_KEY') is None:
    st.error("PINECONE_API_KEY not set. Please set this environment variable and restart the app.")
if os.getenv('PINECONE_ENVIRONMENT') is None:
    st.error("PINECONE_ENVIRONMENT not set. Please set this environment variable and restart the app.")
if os.getenv('PINECONE_INDEX') is None:
    st.error("PINECONE_INDEX not set. Please set this environment variable and restart the app.")
if os.getenv('OPENAI_API_KEY') is None:
    st.error("OPENAI_API_KEY not set. Please set this environment variable and restart the app.")
if os.getenv('EMBEDDINGS_MODEL_NAME') is None:
    st.error("EMBEDDINGS_MODEL_NAME not set. Please set this environment variable and restart the app.")
if os.getenv('GPT3_5_TURBO_MODEL_NAME') is None:
    st.error("GPT3_5_TURBO_MODEL_NAME not set. Please set this environment variable and restart the app.")

st.title("Data Science Tutor Bot 🔎")
query = st.text_input("What do you want to know?")

if st.button("Search"):

    # # get Pinecone API environment variables
    pinecone_api = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
    pinecone_index = os.getenv('PINECONE_INDEX')

    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai_temperature = os.getenv('OPENAI_TEMPERATURE')

    embeddings_model_name = os.getenv('EMBEDDINGS_MODEL_NAME')
    gpt3_5_turbo_model_name = os.getenv('GPT3_5_TURBO_MODEL_NAME')

    text_field = "text"

    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=pinecone_api,
        environment=pinecone_env
    )

    index = pinecone.Index(pinecone_index)
    vector_store = Pinecone(
        index, embeddings.embed_query, text_field
    )

    llm = ChatOpenAI(model_name=gpt3_5_turbo_model_name,
                     temperature=openai_temperature)

    template = """You are a helpful data science tutor chatbot having a conversation with a human.
            Follow exactly these 3 steps:
            1. Read the Context below combining this information with the Question
            2. Answer the Question using only the information in the Context section
            3. Show the source for your answers

            Context : {context}
            User Question: {question}

            The answer and your response should only come from the Context section. If the question is 
            not related to the Context section, politely respond that you are tuned to only answer questions that 
            are related to data science. Use as much detail as possible when responding."""

    prompt = PromptTemplate(
        input_variables=["question", "context"], template=template
    )

    with st.spinner("Summarizing..."):
        try:
            memory = ConversationBufferWindowMemory(k=3,
                                                    memory_key="chat_history",
                                                    input_key="question",
                                                    return_messages=True)

            qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 6}),
                memory=memory,
                # return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": prompt},
                max_tokens_limit=4096
            )

            # Write query answer
            st.markdown("### Answer:")
            st.write(qa.run(query))

        except Exception as e:
            st.error(f"Error with OpenAI Chat Completion: {e}")
