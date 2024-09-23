import sys
import os
from dotenv import load_dotenv
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLineEdit, 
                             QLabel, QListWidget, QMessageBox, QDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")

class CrawlerThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, urls):
        super().__init__()
        self.urls = urls

    def run(self):
        try:
            api_key = os.getenv("FIRECRAWL_API_KEY")
            if not api_key:
                raise ValueError("FIRECRAWL_API_KEY environment variable not set")

            all_docs = []
            for url in self.urls:
                self.update_signal.emit(f"Crawling: {url}")
                loader = FireCrawlLoader(api_key=api_key, url=url, mode="scrape")
                docs = loader.load()
                all_docs.extend(docs)
                self.update_signal.emit(f"Finished crawling: {url}")

            for doc in all_docs:
                for key, value in doc.metadata.items():
                    if isinstance(value, list):
                        doc.metadata[key] = ", ".join(map(str, value))

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            split_docs = text_splitter.split_documents(all_docs)

            self.update_signal.emit(f"Total document chunks: {len(split_docs)}")

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

            urls_hash = hash(tuple(self.urls))
            persistent_directory = os.path.join(db_dir, f"chroma_db_firecrawl_{urls_hash}")
            
            self.update_signal.emit(f"Creating vector store in {persistent_directory}")
            db = Chroma.from_documents(
                split_docs, embeddings, persist_directory=persistent_directory
            )
            self.update_signal.emit(f"Finished creating vector store in {persistent_directory}")

            self.finished_signal.emit(persistent_directory)
        except Exception as e:
            self.update_signal.emit(f"Error: {str(e)}")

class ChatThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, rag_chain, query):
        super().__init__()
        self.rag_chain = rag_chain
        self.query = query

    def run(self):
        try:
            result = self.rag_chain.invoke({"input": self.query, "chat_history": []})
            self.update_signal.emit(f"AI: {result['answer']}")
        except Exception as e:
            self.update_signal.emit(f"Error: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Web Crawler and Chat Application")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # URL input
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("URLs (comma-separated):"))
        self.url_input = QLineEdit()
        url_layout.addWidget(self.url_input)
        main_layout.addLayout(url_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.crawl_button = QPushButton("Crawl Websites")
        self.crawl_button.clicked.connect(self.crawl_websites)
        button_layout.addWidget(self.crawl_button)

        self.load_db_button = QPushButton("Load Existing Database")
        self.load_db_button.clicked.connect(self.load_database)
        button_layout.addWidget(self.load_db_button)
        main_layout.addLayout(button_layout)

        # Status and output
        self.status_output = QTextEdit()
        self.status_output.setReadOnly(True)
        main_layout.addWidget(self.status_output)

        # Chat input and button
        chat_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        chat_layout.addWidget(self.chat_input)
        self.chat_button = QPushButton("Send")
        self.chat_button.clicked.connect(self.send_chat)
        chat_layout.addWidget(self.chat_button)
        main_layout.addLayout(chat_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.rag_chain = None
        self.persistent_directory = None

    def crawl_websites(self):
        urls = [url.strip() for url in self.url_input.text().split(',') if url.strip()]
        if not urls:
            QMessageBox.warning(self, "Error", "Please enter at least one URL.")
            return

        self.crawler_thread = CrawlerThread(urls)
        self.crawler_thread.update_signal.connect(self.update_status)
        self.crawler_thread.finished_signal.connect(self.crawl_finished)
        self.crawler_thread.start()

        self.crawl_button.setEnabled(False)
        self.load_db_button.setEnabled(False)

    def update_status(self, message):
        self.status_output.append(message)

    def crawl_finished(self, persistent_directory):
        self.persistent_directory = persistent_directory
        self.setup_rag_chain()
        self.crawl_button.setEnabled(True)
        self.load_db_button.setEnabled(True)
        self.status_output.append("Crawling finished. You can now start chatting.")

    def load_database(self):
        databases = [d for d in os.listdir(db_dir) if d.startswith("chroma_db_firecrawl_")]
        if not databases:
            QMessageBox.warning(self, "Error", "No existing databases found.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Database")
        dialog.setGeometry(300, 300, 400, 300)

        layout = QVBoxLayout()
        list_widget = QListWidget()
        list_widget.addItems(databases)
        layout.addWidget(list_widget)

        button = QPushButton("Load Selected Database")
        layout.addWidget(button)

        dialog.setLayout(layout)

        def on_button_clicked():
            selected_items = list_widget.selectedItems()
            if selected_items:
                selected_db = selected_items[0].text()
                self.persistent_directory = os.path.join(db_dir, selected_db)
                self.setup_rag_chain()
                self.status_output.append(f"Loaded database: {selected_db}")
                self.status_output.append("You can now start chatting.")
                dialog.accept()
            else:
                QMessageBox.warning(self, "Error", "Please select a database.")

        button.clicked.connect(on_button_clicked)
        dialog.exec()


    def setup_rag_chain(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=self.persistent_directory, embedding_function=embeddings)
        
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given a chat history and the latest user question "
                           "which might reference context in the chat history, "
                           "formulate a standalone question which can be understood "
                           "without the chat history. Do NOT answer the question, just "
                           "reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant for question-answering tasks. Use "
                           "the following pieces of retrieved context to answer the "
                           "question. If you don't know the answer, just say that you "
                           "don't know. Use three sentences maximum and keep the answer "
                           "concise.\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def send_chat(self):
        if not self.rag_chain:
            QMessageBox.warning(self, "Error", "Please crawl websites or load a database first.")
            return

        query = self.chat_input.text()
        if not query:
            return

        self.status_output.append(f"You: {query}")
        self.chat_input.clear()

        self.chat_thread = ChatThread(self.rag_chain, query)
        self.chat_thread.update_signal.connect(self.update_status)
        self.chat_thread.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())