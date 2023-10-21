# Importing libraries
import streamlit as st # library for web app interface creation
from dotenv import load_dotenv # library help to load env files 
import PyPDF2 # library for extracting text from pdf
import os # library to operate on files in operating sytem

from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Qdrant
import qdrant_client
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from template import css,bot_template,user_template # Extract css , and template info from template module
from langchain.llms import HuggingFaceHub


# GLOBAL
# FILEPATH ="Resources\Fundamental_of_Database_Systems.pdf"
FILEPATH = "Resources\sodapdf-converted.pdf"



# Function to extract text from pdfs
def extract_text_from_pdf(pdf_file_path):
    """
    Args:
        pdf_file_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    # using file path we get PDFs
    with open(pdf_file_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text # return all text

# Function to Covert raw text into chunks
def get_text_chunks(text):
    """
    Args:
        text (str): raw text extracted from pdf.
    
    Returns:
        str: chunks of text.
    """
    # Initialize CharacterTextSplitter with user requirement parameters 
    text_splitter =CharacterTextSplitter(
        separator = "\n",
        chunk_size=1000,
        chunk_overlap= 200,
        length_function=len
    )
    # Splitted raw text on given parameters
    chunks = text_splitter.split_text(text)
    # return chunks  
    return chunks


# Storing embedding of text chunks to vectorstore (knowledge base)
def get_vectorstore(text_chunks):
    # create qdrant client
    client =qdrant_client.QdrantClient(
        os.getenv('QDRANT_HOST'),
        api_key = os.getenv('QDRANT_API_KEY')
    )

    # initializing instance of model for embedding from hugging face
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")

    # creating vectors config
    vectors_config = qdrant_client.http.models.VectorParams(
        size=768, # size of vector 
        distance = qdrant_client.http.models.Distance.COSINE
    )

    # create collection
    client.recreate_collection(
        collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
        vectors_config = vectors_config
    )
    
    # initializing vector_store
    vector_store = Qdrant(
        client = client,
        collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
        embeddings=embeddings
    )

    vector_store.add_texts(text_chunks)
    # returning knowledge base
    return vector_store

def get_conversation_chain(vectorstore):
    # instance of model on defined parameters
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})    
    # Conversation buffer memeory is used to extract key information and dynamic conversational interface.
    memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages=True)

    # tuning on it our requirment
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm =llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

def process_user_prompt(UserQuestion):
    # it confgures user previous session state 
    response = st.session_state.conversation({'question': UserQuestion})
    st.session_state.chat_history =response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)


def test():
    st.title("Test Page")
    st.write("This is the Test Page. You can use this page for testing.")


# Main function
def main():
    load_dotenv() # for loading keys from env file  
    st.set_page_config(page_title="Personalized DBMS Learning System", page_icon=":databases:") # Page configuration

    st.write(css, unsafe_allow_html=True)

    # check for session state whether it is initialized or not 
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # check for previous chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= None 

    st.header("DBMS Learning System :") # Header of site 
    user_question = st.text_input("Ask any Question related to DBMS : ") # input label 
    
    # check for user question
    if user_question:
        process_user_prompt(user_question)

    st.write(bot_template.replace("{{MSG}}","Hello! How can I assist you today?"), unsafe_allow_html=True)
    st.write(user_template.replace("{{MSG}}","Wait! i am also waiting for the Prompt.... (Dude tell me the Ques fast??)"), unsafe_allow_html=True)
    with st.sidebar: # initialize sidebar
        st.subheader("WELCOMING YOU TO NEW TECHNOLOGY TUTOR ") # sidebar heading 
        
        if st.button("START LEARNING"): # button to process pdf files
            with st.spinner("Wait"):
                
                # Extracting Text
                raw_text = extract_text_from_pdf(FILEPATH)

                # Create chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Instantiating conversational chain using langchain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()