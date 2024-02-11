import streamlit as st
import warnings

import openai
import PyPDF2  
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Ignore warnings
warnings.filterwarnings("ignore")

# Title
st.title("QA BOT FOR YOUR PDF")
st.write("📖 QA Bot: How can I help you?")

# Add openai api key
# Side bar
with st.sidebar:
    openai_key = st.sidebar.text_input("OpenAI API KEY", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/api-keys)"
    
    st.write("This work is just one of many applications of Retrieval_Augmented Generation (RAG), You can read up on RAG here:")
    "[Retrieval_Augmented Generation](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)"

# Stop if no API key is supplied
if not openai_key:
    st.warning("Enter your OpenAI key to continue")
    st.stop()


# Function to split file into chunks
def chunking(text):
    # Instantiate text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=20,
        length_function=len
    )

    # Apply text_splitter function to text
    pdf_chunks = text_splitter.split_text(text)

    # Create an embeddings for the chunks
    embeddings = OpenAIEmbeddings()

    # Upsert chunks to FAISS vector database
    db_FAISS = FAISS.from_texts(pdf_chunks, embeddings)

    # Return db_faiss
    return db_FAISS


# Function to have a conversation with the pdf file
def main():
    
    
    # Upload a pdf file
    uploaded_file = st.file_uploader("Upload your file", type="pdf")

    # Check if file is uploaded
    if uploaded_file is None:
        st.write("Please upload a pdf file to continue")
        return  # Stop execution

    # Process pdf text
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    
    text = ""
    for page in pdf_reader.pages:
            text += page.extract_text()
        
    # Chunk the text file   
    db_faiss = chunking(text)

    # completion llm
    chatgpt = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.0)

    # Create the QA bot LLM chain
    chain = RetrievalQA.from_chain_type(
        chain_type="stuff",
        llm=chatgpt,
        retriever=db_faiss.as_retriever()
    )
    
    # Session state
    global widget_counter  # Declare widget_counter as global variable

    # Initialize counter for generating unique keys
    widget_counter = 0

    # Loop to continuously prompt for questions
    while True:
        widget_counter += 1  # Increment counter for each widget
        # Accept user query
        query = st.text_input('Enter your question...', key='input_' + str(widget_counter))
        if not query:
            break  # Exit loop if user doesn't ask a question

        # Generate response
        response = chain.run(query)
        
        # Put the query and response in a dict
        qa = {"Query": query,
              "QA Bot": response}
        
        # Display response
        st.write(qa)


if __name__ == "__main__":
    main()

