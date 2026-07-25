from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever          # sparse/keyword-based retriever
from langchain_classic.retrievers import EnsembleRetriever                 # combines multiple retrievers into one
from langchain_core.prompts import PromptTemplate                  # plain text prompt, no chat roles needed
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest", temperature=0.2)

loader = TextLoader("sample_document.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

sparse_retriever = BM25Retriever.from_documents(chunks)
sparse_retriever.k = 3  # how many documents BM25 returns

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
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
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask():
    print("Hybrid Search RAG  - type 'exit' to quit\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer = rag_chain.invoke(question)
        print(f"Bot: {answer}\n")

ask()