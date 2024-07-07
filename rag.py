from langchain import hub
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


def initialize_vectorstore():
    print("Load and process PDF documents...")
    pdf_files = ["Base.pdf", "TechLab Tech4ai.pdf"]
    docs = []

    for pdf_file in pdf_files:
        print(f"Loading PDF file: {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        docs.extend(loader.load())
        print(f"Loaded {pdf_file}")

    print("Split the combined documents..")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)

    print("Create Vectorstore...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
    print("Vectorstore created")
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_tool(query, vectorstore):
    print(f"Executing rag_tool with query: {query}")

    llm = ChatGroq(model="llama3-70b-8192")
    
    retriever = vectorstore.as_retriever()  # Assuming vectorstore is a global variable or accessible

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(query)
    return result

