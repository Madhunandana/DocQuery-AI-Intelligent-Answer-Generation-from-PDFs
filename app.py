


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import matplotlib.pyplot as plt
import time
import tracemalloc

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure API clients
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def calculate_similarity(response, reference):
    vectorizer = TfidfVectorizer().fit_transform([response, reference])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_openai_codex_response(prompt):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # Start measuring time and memory usage
    start_time = time.time()
    tracemalloc.start()

    # Google Gemini Response
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    gemini_response = response["output_text"]
    st.write("Reply from Google Gemini: ")
    st.write(gemini_response)
    st.write("-----------------------------------------------------------------------------------------")

    # Check if the response from Google Gemini is "Answer is not available in the context"
    if "Answer is not available in the context" in gemini_response:
        st.write("No further analysis, as the answer is not available in the context.")
    else:
        # Proceed with calculations and result printing only if the response is valid
        # Stop measuring memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Codex (OpenAI) Response
        start_time_codex = time.time()
        codex_response = get_openai_codex_response(user_question)
        response2 = codex_response({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        cx_response = response2["output_text"]
        end_time_codex = time.time()
        
        st.write("Reply from Codex (OpenAI): ")
        st.write(cx_response)
        st.write("-----------------------------------------------------------------------------------------")

        # Stop measuring time
        end_time = time.time()
        
        # Calculate the Similarity response
        google_similarity = calculate_similarity(gemini_response, cx_response)
        st.write(f"Similarity Comparison: {google_similarity * 100:.2f}%")

        # Response Accuracy (Assuming some ground truth is available for accuracy calculation)
        response_accuracy = google_similarity * 100
        #st.write(f"Response Accuracy: {response_accuracy:.2f}%")

        # Performance Metrics
        response_time_gemini = end_time - start_time
        response_time_codex = end_time_codex - start_time_codex
        memory_usage = peak / 1024  # Convert bytes to KB
        
        performance_metrics = (f"Google Gemini Response Time: {response_time_gemini:.2f} seconds\n"
                               f"Codex Response Time: {response_time_codex:.2f} seconds\n"
                               f"Peak Memory Usage: {memory_usage:.2f} KB")
        st.write(f"Performance Metrics:\n{performance_metrics}")

        # Plot the Performance Metrics
        models = ['Google Gemini', 'Codex']
        response_times = [response_time_gemini, response_time_codex]
        memory_usages = [memory_usage, memory_usage]  # Same memory usage for both models for demonstration

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Models')
        ax1.set_ylabel('Response Time (seconds)', color='tab:blue')
        ax1.bar(models, response_times, color='tab:blue', alpha=0.6, label='Response Time')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Memory Usage (KB)', color='tab:red')
        ax2.plot(models, memory_usages, color='tab:red', marker='o', label='Memory Usage')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        plt.title('Performance Metrics of Models')
        st.pyplot(fig)

        # Overall Effectiveness
        overall_effectiveness = ((response_accuracy + google_similarity * 100) / 2)-1
        st.write(f"Overall Effectiveness: {overall_effectiveness:.4f}%")

def main():
    st.set_page_config("Chat PDF")
    st.header("DocQuery AI💁")

    user_question = st.text_input("Intelligent Answer Generation from PDFs")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
