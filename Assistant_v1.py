import os
from dotenv import load_dotenv
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

def create_vector_store(urls):
    """Crawl the websites, split the content, create embeddings, and persist the vector store."""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")

    all_docs = []
    for url in urls:
        print(f"Begin crawling the website: {url}")
        loader = FireCrawlLoader(api_key=api_key, url=url, mode="scrape")
        docs = loader.load()
        all_docs.extend(docs)
        print(f"Finished crawling {url}")

    for doc in all_docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(all_docs)

    print(f"\nTotal document chunks: {len(split_docs)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    urls_hash = hash(tuple(urls))
    persistent_directory = os.path.join(db_dir, f"chroma_db_firecrawl_{urls_hash}")
    
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(
        split_docs, embeddings, persist_directory=persistent_directory
    )
    print(f"--- Finished creating vector store in {persistent_directory} ---")

    return persistent_directory

def setup_rag_chain(persistent_directory, contextualize_q_system_prompt, qa_system_prompt):
    """Set up the RAG chain with customizable prompts."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    llm = ChatOpenAI(model="gpt-4")

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def continual_chat(rag_chain):
    """Function to simulate a continual chat."""
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

def get_urls():
    """Get multiple URLs from the user."""
    while True:
        url_input = input("Enter URL(s) to crawl (separate multiple URLs with commas, or press Enter to finish): ")
        if not url_input.strip():
            break
        urls = [url.strip() for url in url_input.split(',') if url.strip()]
        if urls:
            return urls
        else:
            print("No valid URLs entered. Please try again.")
    return []

def list_existing_databases():
    """List all existing databases in the db directory."""
    databases = [d for d in os.listdir(db_dir) if d.startswith("chroma_db_firecrawl_")]
    return databases

def select_database():
    """Allow user to select an existing database."""
    databases = list_existing_databases()
    if not databases:
        print("No existing databases found.")
        return None
    print("Existing databases:")
    for i, db in enumerate(databases, 1):
        print(f"{i}. {db}")
    while True:
        choice = input("Enter the number of the database you want to use (or 'q' to quit): ")
        if choice.lower() == 'q':
            return None
        try:
            index = int(choice) - 1
            if 0 <= index < len(databases):
                return os.path.join(db_dir, databases[index])
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")

# def main():
#     while True:
#         print("\nWhat do you want to do?")
#         print("1. Just crawl webpages to store in database")
#         print("2. Crawl webpages and ask questions")
#         print("3. Ask questions (using existing database)")
#         print("4. Exit")
        
#         choice = input("Enter your choice (1-4): ")

#         if choice == '1':
#             urls = get_urls()
#             if urls:
#                 create_vector_store(urls)
#             else:
#                 print("No URLs entered. Returning to main menu.")
#         elif choice == '2':
#             urls = get_urls()
#             if urls:
#                 persistent_directory = create_vector_store(urls)
#                 contextualize_q_system_prompt = input("Enter the contextualize question system prompt (press Enter for default): ") or (
#                     "Given a chat history and the latest user question "
#                     "which might reference context in the chat history, "
#                     "formulate a standalone question which can be understood "
#                     "without the chat history. Do NOT answer the question, just "
#                     "reformulate it if needed and otherwise return it as is."
#                 )
#                 qa_system_prompt = input("Enter the answer question system prompt (press Enter for default): ") or (
#                     "You are an assistant for question-answering tasks. Use "
#                     "the following pieces of retrieved context to answer the "
#                     "question. If you don't know the answer, just say that you "
#                     "don't know. Use three sentences maximum and keep the answer "
#                     "concise."
#                     "\n\n"
#                     "{context}"
#                 )
#                 rag_chain = setup_rag_chain(persistent_directory, contextualize_q_system_prompt, qa_system_prompt)
#                 continual_chat(rag_chain)
#             else:
#                 print("No URLs entered. Returning to main menu.")
#         elif choice == '3':
#             persistent_directory = select_database()
#             if persistent_directory:
#                 contextualize_q_system_prompt = input("Enter the contextualize question system prompt (press Enter for default): ") or (
#                     "Given a chat history and the latest user question "
#                     "which might reference context in the chat history, "
#                     "formulate a standalone question which can be understood "
#                     "without the chat history. Do NOT answer the question, just "
#                     "reformulate it if needed and otherwise return it as is."
#                 )

#                 qa_system_prompt = input("Enter the answer question system prompt (press Enter for default): ") or (
#                     "You are an assistant for question-answering tasks. Use "
#                     "the following pieces of retrieved context to answer the "
#                     "question. If you don't know the answer, just say that you "
#                     "don't know. Use three sentences maximum and keep the answer "
#                     "concise."
#                     "\n\n"
#                     "{context}"
#                 )
#                 rag_chain = setup_rag_chain(persistent_directory, contextualize_q_system_prompt, qa_system_prompt)
#                 continual_chat(rag_chain)
#             else:
#                 print("No database selected. Returning to main menu.")
#         elif choice == '4':
#             print("Exiting the program. Goodbye!")
#             break
#         else:
#             print("Invalid choice. Please try again.")

def main():
    contextualize_q_system_prompt = """
    You are an AI assistant specializing in cloud computing services. Given the chat history and the latest user question, which might reference context in the chat history, formulate a standalone question focused on comparing cloud providers' services in AI, IoT, compute, or database offerings. 

    Key points to consider:
    1. Focus on AWS, Google Cloud, Microsoft Azure, or IBM Cloud.
    2. Emphasize comparisons in virtual machines, managed SQL databases, machine learning, or IoT services.
    3. Highlight key differences in features, performance, or management tools.

    Do NOT answer the question, just reformulate it if needed to focus on these aspects, or return it as is if it already aligns with these criteria.
    """

    qa_system_prompt = """
    You are an expert assistant for comparing cloud service providers, specifically focusing on AWS, Google Cloud, Microsoft Azure, and IBM Cloud. Use the following pieces of retrieved context to answer questions about their services in AI, IoT, compute, and database offerings. 

    When answering:
    1. Provide concise, well-researched comparisons.
    2. Highlight key differences in features, performance, or management tools.
    3. Focus on virtual machines, managed SQL databases, machine learning, or IoT services as relevant.
    4. Keep answers brief and to the point, using three sentences maximum.
    5. If you don't have specific information to compare, state this clearly.

    Your response should be clear, professional, and easily understandable at a glance.

    Context:
    {context}
    """

    while True:
        print("\nWhat do you want to do?")
        print("1. Just crawl webpages to store in database")
        print("2. Crawl webpages and ask questions")
        print("3. Ask questions (using existing database)")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            urls = get_urls()
            if urls:
                create_vector_store(urls)
            else:
                print("No URLs entered. Returning to main menu.")
        elif choice == '2':
            urls = get_urls()
            if urls:
                persistent_directory = create_vector_store(urls)
                rag_chain = setup_rag_chain(persistent_directory, contextualize_q_system_prompt, qa_system_prompt)
                continual_chat(rag_chain)
            else:
                print("No URLs entered. Returning to main menu.")
        elif choice == '3':
            persistent_directory = select_database()
            if persistent_directory:
                rag_chain = setup_rag_chain(persistent_directory, contextualize_q_system_prompt, qa_system_prompt)
                continual_chat(rag_chain)
            else:
                print("No database selected. Returning to main menu.")
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()