import os
from io import BytesIO
import json
import queue
import sys
import time

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import vosk
import sounddevice as sd
import pyttsx3

load_dotenv()
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Vosk offline speech recognition setup ---
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def recognize_speech_offline(model_path):
    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, 16000)

    print("Speak now... (listening for 10 seconds max)")

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        start_time = time.time()
        while True:
            if time.time() - start_time > 10:  # timeout after 10 sec
                print("Listening timeout.")
                return None
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text = json.loads(result).get('text', '')
                if text:
                    print(f"You said: {text}")
                    return text

# --- PDF Reading ---
def get_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

# --- Text chunking ---
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# --- Vector store ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')
    return vector_store

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# --- QA Chain ---
def get_conversational_chain():
    prompt_template = """
Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
If there is no answer, then just say "Answer is not available in the context." Don't provide wrong answers.

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- Text-to-Speech ---
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# --- Main conversation loop ---
def main():
    print("Welcome to Chat with PDF + Offline Voice Q&A!")
    pdf_path = input("Please enter the path to your PDF file: ").strip()
    if not os.path.isfile(pdf_path):
        print("File not found. Exiting.")
        return

    print("Reading PDF...")
    raw_text = get_pdf_text(pdf_path)
    print("Splitting text into chunks...")
    text_chunks = get_text_chunks(raw_text)
    print("Creating vector store (this may take a moment)...")
    get_vector_store(text_chunks)
    print("Vector store saved locally.")

    vector_store = load_vector_store()
    chain = get_conversational_chain()

    while True:
        print("\nYou can ask questions about your PDF now!")
        print("Say 'exit' to quit.")
        model_path = "D:/Dev/talk_to_pdf_with_gemini/Chat-with-PDF/vosk-model-small-en-us-0.15"  # CHANGE THIS to your actual model path
        question = recognize_speech_offline(model_path)
        if question is None:
            print("No speech detected. Try again.")
            continue
        if question.lower() in ["exit", "quit", "stop"]:
            print("Exiting. Goodbye!")
            break

        docs = vector_store.similarity_search(question)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        answer = response["output_text"]
        print(f"Answer:\n{answer}")
        speak_text(answer)

if __name__ == "__main__":
    main()
