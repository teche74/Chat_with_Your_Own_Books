# Chat With Your Own Books

**Chat_with_Your_Own_Books** is a web application that allows users to interact with a chatbot to get personalized information and answers related to Database Management Systems (DBMS). This project leverages natural language processing and machine learning to provide a dynamic learning experience.

## Table of Contents
- [Features](#features)
- [Pipeline](#pipeline)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- Interactive Chatbot: Users can ask questions and receive answers related to DBMS.
- PDF Text Extraction: The application extracts text from a PDF document for answering user queries.
- Language Models: Utilizes language models to understand and respond to user queries.
- Personalized Learning: Provides a dynamic learning experience with personalized responses.


## Pipeline

![WhatsApp Image 2023-10-02 at 06 41 42_e07c950b](https://github.com/teche74/Chat_with_Your_Own_Books/assets/129526047/ea474a49-3758-4321-bdbb-325b52c6de28)


check app at https://chatwithyourownbooks-j4oxdmfvq9wt86taesobck.streamlit.app/

## Getting Started

Follow the steps below to set up and run this project locally on your machine.

### Prerequisites

- Python (version 3.6 or higher)
- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [qdrant-client](https://pypi.org/project/qdrant-client/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/teche74/Chat_with_Your_Own_Books.git

2. Change the directory to the project folder:
   ```bash
   cd chat-with-your-own-books

4. Install the required Python packages:
    ```bash
    pip install -r requirements.txt

5. Set up your environment variables by creating a .env file with the necessary configuration (API keys, file paths, etc.).

6. Run the application using Streamlit:
    ```bash
    streamlit run main.py

## Usage
Open a web browser and go to the URL provided by Streamlit.
Ask questions related to Database Management Systems (DBMS) in the input field.
Interact with the chatbot and receive responses.


## Contributing
If you'd like to contribute to this project, please follow these guidelines:

- Fork the repository on GitHub.
- Create a new branch with a descriptive name for your feature or bug fix.
- Make your changes and test them thoroughly.
- Commit your changes with clear and concise messages.
- Push your changes to your fork.
- Create a pull request to the original repository, describing your changes.


## License
This project is licensed under the MIT License.
