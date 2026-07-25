from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest", temperature=0.2)

loader = TextLoader("sample_document.txt")
documents = loader.load()
 
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
 
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,             
        "fetch_k": 10,      
    }
)

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
 
prompt = PromptTemplate.from_template(
    "Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

rag_chain = (
    {"context": mmr_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask():
    print("===== MMR RAG =====\n")
 
    while True:
        question = input("You: ")
 
        if question.lower() == "exit":
            print("Goodbye!")
            break
 
        answer = rag_chain.invoke(question)
        print(f"Bot: {answer}\n")
 

ask()