import openai
import speech_recognition as sr
import subprocess
from pydub import AudioSegment
from pathlib import Path
import requests
import os
from dotenv import load_dotenv,find_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import OpenAI
from io import BytesIO
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from dotenv import load_dotenv,find_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import wave

load_dotenv(find_dotenv())

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")



def recognize_wav(file_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = r.record(source)
            user_input = r.recognize_google(audio_data, language='vi-VN')
            return user_input
            # Uncomment the following lines if you want to use the whisper function and check_quit function
            # if check_quit(user_input):
            #     return True
    except sr.UnknownValueError:
        return None

# Define the OpenAI API key
openai.api_key = "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"

def generate_response(messages):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=messages,
        temperature=0.8
    )

    response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    print(f"\n{response}\n")
    return messages


def text_to_speech2(text):
    model_path = "/Users/phuongminh/Downloads/Sơn/Project/model.onnx"
    output_file = "/Users/phuongminh/Downloads/Sơn/Project/Voice_Test/Test.wav"
    config_path = "/Users/phuongminh/Downloads/Sơn/Project/config.json"
    webm_file_path = "temp_audio_play.mp3"
    command = f"echo '{text}' | piper -m {model_path} --output_file {output_file} --config {config_path}"
    subprocess.run(command, shell=True)

        # Load the WAV audio file
    audio = AudioSegment.from_wav(output_file)

        # Export the audio to MP3 format
    audio.export(webm_file_path, format="mp3")

    return webm_file_path



def get_answer_from_pdf(pdf_input, question):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
 
    try:
        # Attempt to read the input as a file path
        with open(pdf_input, 'rb') as file:
            pdf_data = file.read()
    except TypeError:
        # If the input is not a valid file path, assume it's a file-like object
        pdf_data = pdf_input.read()

        # Create a BytesIO object from the PDF data
    pdf_bytes = BytesIO(pdf_data)

    # Create a PdfReader instance from the BytesIO object
    doc_reader = PdfReader(pdf_bytes)

    text = ''
    for page in doc_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(text)

    # Create embeddings and FAISS index
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    # Set up retriever and QA chain
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=OpenAI(api_key=api_key), chain_type="stuff", retriever=retriever,
                                     return_source_documents=True)

    # Get answer from the QA chain
    result = qa({"query": question})
    answer = result["result"]

    return answer


vector_db_path = "vectorbase/db_faiss"


def embed_pdf_text2(pdf_path):
    # Load the PDF file
    doc_reader = PdfReader(pdf_path)

    # Extract text from each page
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Create embeddings for the text chunks
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    docsearch.save_local(vector_db_path)

    return docsearch

embed_pdf_text2("data/Vietnam-Travel-đã-gộp.pdf")
def qa(query):
    query_strings = [str(msg) for msg in query]

    query_string = "\n".join(query_strings)

    # Set up FAISS as a generic retriever
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Create the chain to answer questions
    rqa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0),
                                      chain_type="stuff",
                                      retriever=retriever,
                                      return_source_documents=True)

    # Get the result and print it
    result = rqa(query_string)
    print(result['result'])



def qa2(messages):
    system_message = [{"role": "system", "content": "You are a helpful Tourism AI chatbot, that answers questions about the tourism industry asked by User using Vietnamese."}]
    messages = system_message + messages

    # Convert messages to a single string
    query_strings = [msg['content'] for msg in messages]
    query_string = "\n".join(query_strings)

    # Set up FAISS as a generic retriever
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Create the chain to answer questions
    rqa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0),
                                      chain_type="stuff",
                                      retriever=retriever,
                                      return_source_documents=True)

    # Get the result
    result = rqa(query_string)

    response = {
        "choices": [
            {
                "message": {
                    "content": result['result'],
                    "role": "assistant"
                }
            }
        ]
    }

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

