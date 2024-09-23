import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain import hub

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

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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

def setup_rag_chain(persistent_directory):
    """Set up the RAG chain with improved prompts and error handling."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    c_q_prompt = """
    You are an AI assistant specializing in cloud computing services. Given the chat history and the latest user question, which might reference context in the chat history, formulate a standalone question focused on comparing cloud providers' services in AI, IoT, compute, or database offerings. 

    Key points to consider:
    1. Focus on AWS, Google Cloud, Microsoft Azure, or IBM Cloud.
    2. Emphasize comparisons in virtual machines, managed SQL databases, machine learning, or IoT services.
    3. Highlight key differences in features, performance, or management tools.

    Do NOT answer the question, just reformulate it if needed to focus on these aspects, or return it as is if it already aligns with these criteria.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", c_q_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    q_s_prompt = """
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

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", q_s_prompt),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def create_the_react_agent(rag_chain):
    """Create a ReAct agent using the RAG chain with error handling and custom prompts."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def rag_tool(query, chat_history=None):
        try:
            input_data = {
                "input": query,
                "context": "You are comparing cloud services like RDS.",
                "chat_history": chat_history or []
            }
            result = rag_chain.invoke(input_data)
            if isinstance(result, dict) and 'answer' in result:
                return result['answer']
            elif isinstance(result, str):
                return result
            else:
                return "I couldn't find a relevant answer. Can you please rephrase your question?"
        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}. Please try again or rephrase your question."

    tools = [
        Tool(
            name="RAG_Chain",
            func=rag_tool,
            description="Use this tool to get detailed information about cloud providers' services, including comparisons between different providers."
        )
    ]
    
    prompt = hub.pull("firecrawltocompare/bytebirdie")

    agent = create_react_agent(
        llm=llm, 
        tools=tools, 
        prompt=prompt,
    )
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )


def continual_chat(agent):
    """Function to simulate a continual chat using the ReAct agent with improved user experience."""
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    print("Type 'help' for additional commands.")
    
    while True:
        query = input("You: ")
        
        if query.lower() == 'exit':
            print("Ending the conversation. Goodbye!")
            break
        elif query.lower() == 'help':
            print("Available commands:")
            print("- 'exit': End the conversation")
            print("- 'clear': Clear the conversation (for display purposes only)")
            print("- 'help': Show this help message")
            continue
        elif query.lower() == 'clear':
            print("\n" * 50)  # Print newlines to simulate clearing the screen
            print("Conversation display cleared.")
            continue
        
        start_time = time.time()
        
        try:
            response = agent.invoke({"input": query})
            response = response['output']
        except Exception as e:
            response = f"An error occurred: {str(e)}. Please try again or rephrase your question."
        
        end_time = time.time()
        
        print(f"AI: {response}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print()

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

def main():
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
                rag_chain = setup_rag_chain(persistent_directory)
                agent_executor = create_the_react_agent(rag_chain)
                continual_chat(agent_executor)
            else:
                print("No URLs entered. Returning to main menu.")
        elif choice == '3':
            persistent_directory = select_database()
            if persistent_directory:
                rag_chain = setup_rag_chain(persistent_directory)
                agent_executor = create_the_react_agent(rag_chain)
                continual_chat(agent_executor)
            else:
                print("No database selected. Returning to main menu.")
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()