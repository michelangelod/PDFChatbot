import streamlit as st
import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.chains import  RetrievalQA
import os

openai_api_key = st.text_input(label="Your Openai API Key...")
os.environ['OPENAI_API_KEY'] = openai_api_key

def read_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        num_pages = reader.getNumPages()
        text = ""
        for page_num in range(num_pages):
            page = reader.getPage(page_num)
            text += page.extractText()
    return text

st.title("PDF Chatbot")
uploaded_file = st.file_uploader("SELECT A PDF FILE", type=["pdf"])

if uploaded_file is not None and openai_api_key != "":
    file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(file_details)

    # 保存 PDF 文件到临时路径
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_path = "temp.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    # turn docs into embeddings (vector space)
    try:
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(temperature=0.0)
        db = DocArrayInMemorySearch.from_documents(
            docs,
            embeddings,
        )
        retriever = db.as_retriever()
        qa_map_reduce = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce", 
            retriever = retriever, # 检索器，
            verbose = True,
        )
    except:
        st.warning("Please Enter A Valid Key")

question = st.text_input("Question...")

if openai_api_key == "":
    st.warning("Please Enter Your Openai Api key")

elif question != "":
    try:
        answer = qa_map_reduce.run(question)
        st.write(answer)
    except:
        st.warning("Please Recheck Your Openai Api key")

