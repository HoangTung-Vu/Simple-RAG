import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def load_and_split_pdf(file_path, chunk_size=1000):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    return text_splitter.split_documents(data)

def initialize_embedding_model(model="models/text-embedding-004"):
    return GoogleGenerativeAIEmbeddings(model=model)

def get_vector_store(docs=None, embedding_model=None, persist_dir="chroma_db"):
    if os.path.exists(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        #vectorstore.persist()
        return vectorstore

def initialize_llm(model="gemini-1.5-pro", temperature=0.3, max_tokens=5000):
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_tokens=max_tokens)

# Function to create RAG chain
def create_rag_chain(retriever, llm, system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Main function to run the app
def run_rag_app(question, persist_dir="chroma_db", file_path=None):
    # Initialize embedding model
    embedding_model = initialize_embedding_model()
    
    # Load or create vector store
    if file_path and not os.path.exists(persist_dir):
        docs = load_and_split_pdf(file_path)
        vectorstore = get_vector_store(docs, embedding_model, persist_dir)
    else:
        vectorstore = get_vector_store(embedding_model=embedding_model, persist_dir=persist_dir)
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Initialize LLM and RAG chain
    system_prompt = (
        "Bạn là một chuyên gia y khoa người Việt Nam với kiến thức sâu rộng về y khoa. "
        "Nhiệm vụ của bạn là trả lời các câu hỏi dựa trên các tài liệu y khoa được cung cấp. "
        "Tài liệu trên có nhiều nhiễu lỗi ký tự, hãy tự khử chúng một cách hợp lý."
        "Nếu câu trả lời không được nêu rõ ràng trong các tài liệu, hãy nói rằng bạn không biết. "
        "Câu trả lời của bạn cần rõ ràng, chính xác.\n\n"
        "{context}"
    )
    llm = initialize_llm()
    rag_chain = create_rag_chain(retriever, llm, system_prompt)

    # Invoke the RAG chain with the input question
    response = rag_chain.invoke({"input": question})
    return response['answer']

# Example usage
if __name__ == "__main__":
    # First time: Create and save vectorstore
    file_path = "/home/hoangtung/Code/RAG/documents/hoasinh1.pdf"
    question = "Sterol là gì?"
    response = run_rag_app(question, file_path=file_path)
    print(response)
    
    # Subsequent times: Load existing vectorstore
    question = "Câu hỏi khác?"
    response = run_rag_app(question)
    print(response)