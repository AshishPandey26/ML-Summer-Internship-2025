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

import speech_recognition as sr
import pyttsx3


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
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
    If there is no answer, then just say "Answer is not available in the context." Don't provide wrong answers.
    
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
    speak_text(answer)

# ----
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        # st.info("Listening... Speak now!")
        st.info("Get ready to speak in 3 seconds...")
        time.sleep(3)

        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            st.error("Listening timed out while waiting for phrase to start.")
        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
        except sr.RequestError:
            st.error("Speech Recognition API unavailable.")
        return None
    
    

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.say(text)
    engine.runAndWait()




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

    input_mode = st.radio("Select input mode:", ["Type", "Speak"])

    user_question = ""
    if input_mode == "Type":
        user_question = st.text_input("Type your question below:")
    elif input_mode == "Speak":
        if st.button("🎤 Speak Now"):
            user_question = recognize_speech_from_mic() or ""
    if user_question:
        user_input(user_question, speak=(input_mode == "Speak"))

if __name__ == "__main__":
    main()
