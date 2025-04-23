import os
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

# Chargement des variables d'environnement
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Préparation du corpus
loader = TextLoader('/content/drive/MyDrive/regetude.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

# Embeddings et index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.from_documents(docs, embeddings)

# LLM et prompt
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
)

template = """
Tu es un assistant pédagogique francophone de l'INSA de Toulouse. Tu réponds toujours en français, même si la question est posée dans une autre langue.
Tu peux répondre aussi bien à des questions pédagogiques qu'à des questions de conversation générale comme "ça va ?", "tu fais quoi ?", etc.
Utilise le contexte ci-dessous si nécessaire pour répondre à la question. Si tu ne sais pas, dis-le simplement.
Ta réponse doit être concise, naturelle, et tenir en 2 phrases maximum.

Contexte : {context}
Question : {question}
Réponse :
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain_prompt = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
qa_chain = RetrievalQA(retriever=faiss_index.as_retriever(), combine_documents_chain=qa_chain_prompt)

# Interface Streamlit
st.title("Assistant pédagogique INSA ✨")
question = st.text_input("Pose une question 👇")
if question:
    result = qa_chain({"query": question})
    st.write("**Réponse :**", result["result"])
