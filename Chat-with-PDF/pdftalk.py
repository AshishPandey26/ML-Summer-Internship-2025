import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import speech_recognition as sr
import pyttsx3

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Hide tkinter root window
def select_pdf_file():
    Tk().withdraw()
    file_path = askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    return file_path

def get_pdf_text(file_path):
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
    If there is no answer, then just say "Answer is not available in the context." Don't provide wrong answers.
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("🎤 Get ready to speak in 3 seconds...")
        time.sleep(3)
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("⏰ Timeout: Didn't hear anything.")
        except sr.UnknownValueError:
            print("🤷 Couldn't understand the audio.")
        except sr.RequestError:
            print("⚠️ Speech Recognition API unavailable.")
    return None

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

def answer_question(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    answer = response["output_text"]
    print("\n🧞 Answer:", answer)
    speak_text(answer)

def main():
    print("📄 Select a PDF file to chat with:")
    file_path = select_pdf_file()
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    print("⏳ Processing your PDF...")
    raw_text = get_pdf_text(file_path)
    chunks = get_text_chunks(raw_text)
    get_vector_store(chunks)
    print("✅ PDF processed! Ask your question via speech now.\n")

    while True:
        print("\n🎙️ Ready for your question. Say 'quit' to exit.")
        question = recognize_speech_from_mic()
        if question:
            if "quit" in question.lower():
                print("👋 Exiting. Bye!")
                break
            answer_question(question)

if __name__ == "__main__":
    main()
