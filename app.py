# Importing libraries
import PyPDF2 
import requests
import random
import streamlit as st # library for web app interface creation
from dotenv import load_dotenv # library help to load env files 
from PyPDF2 import PdfReader# library for extracting text from pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from template import css,bot_template,user_template     # Extract css , and template info from template module
from langchain.llms import HuggingFaceHub
import os
import time 
from langchain.document_loaders.csv_loader import CSVLoader


# Global paths define 
FILEPATH ="Resources\sampfin.pdf"
PERDIR = "Embeddings"


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
    text_splitter =RecursiveCharacterTextSplitter(
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
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # instance of model on defined parameters
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    # Conversation buffer memeory is used to extract key information and dynamic conversational interface.
    memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages=True)

    # tuning on it our requirment
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm =llm,
        retriever=vectorstore.as_retriever(search_kwargs ={"k":2}),
        memory=memory
    )

    return conversation_chain

def process_user_prompt(UserQuestion):
    if st.session_state.conversation is not None:
        # it confgures user previous session state 
        response = st.session_state.conversation({'question': UserQuestion})
        st.session_state.chat_history =response['chat_history']

        # Reverse the order of messages before displaying
        reversed_history = reversed(st.session_state.chat_history)

        for i, message in enumerate(reversed_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Please start learning before asking a question.")

def generate_test_question(topic, difficulty, amount_of_questions, test_type):
    url = "https://chatgpt-42.p.rapidapi.com/gpt4"

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"Generate a {test_type} based question for {topic} with {difficulty}difficulty. Generate exactly {amount_of_questions} questions in python list format"
            }
        ],
        "tone": "Balanced"
    }

    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "d05f815ffcmsh5492e769dc370cep174791jsnfb20cae30abd",
        "X-RapidAPI-Host": "chatgpt-42.p.rapidapi.com"
    }

    try:
        # Make a request to the GPT-4 API for generating a question
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            # Extract the generated question from the API response
            question_data = response.json()
            question = question_data.get("choices", [{"message": "No question found."}])[0]["message"]
            return question
        else:
            return f"Error generating question. Status code: {response.status_code}"

    except Exception as e:
        return f"An error occurred: {str(e)}"

def test_mark(question, correct_answer):
    st.write(f"**Q: {question}**")
    user_answer = st.text_input("Your Answer:")
    st.write(f"Your Answer: {user_answer}")

        # Check correctness automatically
    if user_answer.lower() == correct_answer.lower():
        st.success("Correct! Well done.")
    else:
        st.error(f"Incorrect. The correct answer is: {correct_answer}")

def test_modal():
    with st.form(key='test_form'):
        st.subheader("Test Settings")
        topic = st.selectbox("Enter the topic for the test:", ["Introduction to DBMS", "SQL", "Normalization"])
        difficulty = st.selectbox("Select Difficulty", ["easy", "medium", "hard"])
        amount_of_questions = st.slider("Number of Questions", min_value=1, max_value=20, value=5)
        test_type = st.selectbox("Test Type", ["topic based", "mcq", "SQL commands"])
        if test_type == "topic based":
            test_type = st.text_input("topic name : ")

        # Use the button with a specific key to avoid unnecessary reruns
        take_test_button = st.form_submit_button("TAKE TEST")
        
        if take_test_button:
            st.session_state.show_more_info = True  # Set show_more_info to True when the form is submitted
            with st.spinner("GENERATING QUES"):
                for _ in range(amount_of_questions):
                    question, correct_answer = generate_test_question(topic, difficulty, amount_of_questions, test_type)
                    test_mark(question, correct_answer)
                st.write("### Test Completed!")
                st.session_state.test_completed = True

    if st.session_state.test_completed:
        st.experimental_rerun()

    

def animate_robot():
    animation_html = """
    <style>
        @keyframes floatCloud {
            0%, 100% {
                transform: translateY(0);
            }s
            50% {
                transform: translateY(-10px);
            }
        }

        @keyframes rotateBook {
            0% {
                transform: rotate(0);
            }
            100% {
                transform: rotate(360deg);
            }
        }

        @keyframes bounceQuestionMark {
            0%, 70%, 100% {
                transform: translateY(0);
            }
            35% {
                transform: translateY(-20px);
            }
        }

        .front-page-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            color: #fff;
            overflow: hidden;
        }

        .cloud-container {
            position: relative;
            height: 100px;
            width: 100%;
            overflow: hidden;
            animation: floatCloud 4s infinite;
        }

        .cloud {
            font-size: 3em;
            color: #fff;
            position: absolute;
        }

        .book {
            font-size: 4em;
            color: #fff;
            animation: rotateBook 3s infinite;
            margin-top: 20px;
        }

        .question-mark {
            font-size: 4em;
            color: #fff;
            animation: bounceQuestionMark 2s infinite;
            margin-top: 20px;
        }

        .welcome-message {
            margin-top: 20px;
            font-size: 2em;
        }
        .flash-card {
            font-size: 2em;
            color: #fff;
            animation: fadeIn 1s;
            margin-top: 20px;
            border: 2px solid #fff;
            padding: 20px;
            border-radius: 10px;
        }
    </style>

    <div class="front-page-container">
        <div class="cloud-container">
            <div class="cloud">&#9729;</div>
            <div class="cloud">&#9729;</div>
            <div class="cloud">&#9729;</div>
        </div>
        <div class="book">&#128214;</div>
        <div class="question-mark">&#10067;</div>
        <div class="welcome-message">Welcome to Personalized DBMS Learning System</div>
    </div>
    """

    st.markdown(animation_html, unsafe_allow_html=True)


def get_random_question():
    st.header("DBMS Interactive Quiz :rocket:")  # Header of the site
    questions = [
        {"question": "What is a foreign key in a relational database?", "options": ["A key from another table", "A unique identifier for a table", "A primary key in another database", "A key used for sorting data"], "correct_answer": "A key from another table"},
        {"question": "Which SQL clause is used to sort the result set?", "options": ["ORDER BY", "GROUP BY", "HAVING", "WHERE"], "correct_answer": "ORDER BY"},
        {"question": "What is the purpose of the GROUP BY clause in SQL?", "options": ["Group rows based on a column's values", "Filter rows based on conditions", "Sort the result set", "Join multiple tables"], "correct_answer": "Group rows based on a column's values"},
        {"question": "Which of the following is an example of a NoSQL database?", "options": ["MongoDB", "MySQL", "Oracle Database", "SQLite"], "correct_answer": "MongoDB"},
        {"question": "In the context of databases, what is denormalization?", "options": ["Introducing redundancy to improve query performance", "Organizing data to reduce redundancy", "Removing data from a database", "Encrypting data in a database"], "correct_answer": "Introducing redundancy to improve query performance"},
        {"question": "What is a stored procedure in a database?", "options": ["A precompiled set of one or more SQL statements", "A table that stores data temporarily", "A way to retrieve data from multiple tables", "A unique identifier for a record"], "correct_answer": "A precompiled set of one or more SQL statements"},
        {"question": "Which SQL command is used to delete data from a database?", "options": ["REMOVE", "DELETE", "DROP", "ERASE"], "correct_answer": "DELETE"},
        {"question": "What is a view in a database?", "options": ["A virtual table based on the result of a SELECT query", "A table that stores data temporarily", "A way to retrieve data from multiple tables", "A unique identifier for a record"], "correct_answer": "A virtual table based on the result of a SELECT query"},
        {"question": "What is the purpose of the HAVING clause in SQL?", "options": ["Filter rows based on conditions", "Sort the result set", "Group rows based on a column's values", "Filter aggregated data"], "correct_answer": "Filter aggregated data"},
        {"question": "Which type of join returns all rows when there is a match in one of the tables?", "options": ["LEFT JOIN", "INNER JOIN", "RIGHT JOIN", "OUTER JOIN"], "correct_answer": "OUTER JOIN"},
        {"question": "What is the role of a database index?", "options": ["Improve query performance by speeding up data retrieval", "Store data in a structured format", "Create a backup of the database", "Sort the data in the database"], "correct_answer": "Improve query performance by speeding up data retrieval"},
        {"question": "Which SQL function is used to find the maximum value in a column?", "options": ["MAX()", "MIN()", "SUM()", "AVG()"], "correct_answer": "MAX()"},
        {"question": "What is a trigger in the context of databases?", "options": ["A set of instructions that automatically execute in response to a certain event", "A way to retrieve data from multiple tables", "A virtual table based on the result of a SELECT query", "A table that stores data temporarily"], "correct_answer": "A set of instructions that automatically execute in response to a certain event"},
        {"question": "What is a primary key in a relational database?", "options": ["A unique identifier for a table", "A key from another table", "A key used for sorting data", "An index on all columns"], "correct_answer": "A unique identifier for a table"},
        {"question": "In SQL, what is the purpose of the DISTINCT keyword?", "options": ["Retrieve unique values from a column", "Sort the result set", "Filter rows based on conditions", "Group rows based on a column's values"], "correct_answer": "Retrieve unique values from a column"},
        {"question": "Which SQL function is used to count the number of rows in a table?", "options": ["COUNT()", "SUM()", "AVG()", "MAX()"], "correct_answer": "COUNT()"},
        {"question": "What is the purpose of the UNION operator in SQL?", "options": ["Combine the result sets of two or more SELECT statements", "Filter rows based on conditions", "Sort the result set", "Join multiple tables"], "correct_answer": "Combine the result sets of two or more SELECT statements"},
        {"question": "Which type of database model organizes data in a tree-like structure?", "options": ["Hierarchical Database", "Relational Database", "NoSQL Database", "Object-Oriented Database"], "correct_answer": "Hierarchical Database"},
        {"question": "What is the role of the COMMIT statement in a database transaction?", "options": ["Save the changes made during the transaction", "Roll back the changes made during the transaction", "Start a new transaction", "Lock the database for exclusive access"], "correct_answer": "Save the changes made during the transaction"},
        {"question": "Which SQL command is used to add a new row to a table?", "options": ["ADD", "INSERT", "UPDATE", "CREATE"], "correct_answer": "INSERT"},
        {"question": "What is the purpose of the ROLLBACK statement in a database transaction?", "options": ["Undo the changes made during the transaction", "Commit the changes made during the transaction", "Retrieve data from the database", "Sort the data in the database"], "correct_answer": "Undo the changes made during the transaction"},
        {"question": "In a relational database, what is the role of the JOIN operation?", "options": ["Combine rows from two or more tables based on a related column", "Filter rows based on conditions", "Sort the result set", "Group rows based on a column's values"], "correct_answer": "Combine rows from two or more tables based on a related column"},
        {"question": "What is the purpose of the AVG() function in SQL?", "options": ["Calculate the average value of a numeric column", "Retrieve the maximum value in a column", "Count the number of rows in a table", "Sort the result set"], "correct_answer": "Calculate the average value of a numeric column"}
    ]

    current_question = random.choice(questions)
    correct_answer = current_question["correct_answer"]

    st.markdown('<div class="flash-card" style="text-align: center;">{}</div>'.format(current_question['question']), unsafe_allow_html=True)

    user_answer_index = st.radio("Select the correct option:", range(len(current_question["options"])), format_func=lambda i: current_question["options"][i])
    
    if "user_answer" not in st.session_state:
        st.session_state.user_answer = None

    if st.session_state.user_answer is None:
        st.session_state.user_answer = current_question["options"][user_answer_index]

        # Display options
        option_container = st.empty()
        option_container.markdown(
            '<div class="option-container">{}</div>'.format(
                ''.join(
                    '<div class="option" onclick="this.previousSibling.checked=true;">{}</div>'.format(opt)
                    for opt in current_question["options"]
                )
            ),
            unsafe_allow_html=True
        )    
    
    if st.button("Submit Answer"):
        if st.session_state.user_answer.lower() == correct_answer.lower():
            st.write(f'<div class="success-message">Correct! Well done.</div>', unsafe_allow_html=True)
        else:
            st.write(f'<div class="error-message">Incorrect. The correct answer is: {correct_answer}</div>', unsafe_allow_html=True)    
    if st.button("Next Question"):
        st.session_state.user_answer = None
        st.experimental_rerun()


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

    
    navigation = st.sidebar.radio("Navigation", ["Home", "Start Learning", "Take a Test"])
    if navigation == "Start Learning":
        st.header("DBMS Learning System :") # Header of site 
        user_question = st.text_input("Ask any Question related to DBMS : ") # input label 
    
        # check for user question
        if user_question:
            process_user_prompt(user_question)

        st.write(bot_template.replace("{{MSG}}","Hello! How can I assist you today?"), unsafe_allow_html=True)
        st.write(user_template.replace("{{MSG}}","Wait! i am also waiting for the Prompt.... (Dude tell me the Ques fast??)"), unsafe_allow_html=True)
        
        if st.button("START LEARNING"): # button to process pdf files
            with st.spinner("Wait"):
                
                # get pdf text
                raw_text = extract_text_from_pdf(FILEPATH)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
        
    elif navigation == "Take a Test":
        if st.button("Take Test"):  # This button will trigger a rerun when clicked
            signal = test_modal() 
            
    elif navigation == "Home":
        animate_robot()
        get_random_question()


if __name__ == '__main__':
    main()
