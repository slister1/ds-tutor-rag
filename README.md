Project Overview

Are LLM-based Chatbots Reliable and Safe Enough to Provide Free Data Science Tutoring for Marginalized Communities
In collaboration with Jonathan Hodges, Edward Shin, and Neil Fong for Applied Machine Learning course (DATASCI W207). 

Installation Instructions

The first step is breaking up the textbook dataset into text chunks and storing as embeddings in the Pinecone vector database. This is what code/indexer.py takes care of. This will require the textbook dataset of 45 PDFs in the 'data' directory. This wasn't added to github due to large size. Open AI and Pinecone accounts and API Keys are required as well. These need to be set with the following environment variables.  <br>
  <br>
os.environ["OPENAI_API_KEY"] =   <br>
os.environ["PINECONE_API_KEY"] =   <br>
os.environ["PINECONE_INDEX"] = data-science-textbooks  <br>
os.environ["EMBEDDINGS_MODEL_NAME"] = text-embedding-ada-002 <br> 
  <br>
Once the text embeddings are persisted to Pinecone, the chatbot can be tested either in the Streamlit app (chat_app.py) or notebook (Generate_LLM_Answers.ipynb)  <br>
  <br>
streamlit run chat_app.py   <br>
or you can view the demo video in the slides folder  <br>
  <br>
There is another README.md file in the vicuna-eval directory that explains the evaluation approach and the JSONL files in the directory.  <br>
