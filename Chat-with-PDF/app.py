import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_file = BytesIO(pdf.read())  # read the uploaded file
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')

def get_conversational_chain():
    prompt_template = """
    
    analyze the pdf content and create a mock exam paper with 60 multiple choice questions (MCQs) based on the content.
    Analyze the priority of the content and chapters assigned by the user and then create the mock paper according to the priority from low to high.
    Also assign a difficulty level to each question (easy, medium, hard) based on the content.
    Also assign marks to each question (3, 4, 5) based on the difficulty level.
    Provide the ansswer key for the questions at the end of the mock paper.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    answer = response["output_text"]
    st.write("Reply:", answer)

def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using Gemini🧞‍♂️")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Documents processed. You can now ask questions!")

    st.subheader("Ask your Question")
    user_question = st.text_input("Type your question below:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()