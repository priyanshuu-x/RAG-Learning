from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest", temperature=0.2)

loader = TextLoader("sample_document.txt")
documents = loader.load()

semantic_splitter = SemanticChunker(
    embeddings,                              
    breakpoint_threshold_type="percentile",   
    breakpoint_threshold_amount=90            
)

chunks = semantic_splitter.split_documents(documents)
print(f"Document split into {len(chunks)} semantic chunk(s).\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} ---")
    print(chunk.page_content)
    print()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the context provided below. "
               "If the answer isn't in the context, say you don't know."),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask():
    print("---- Semantic Chunking ----- ( type 'exit' to quit)\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer = rag_chain.invoke(question)
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    ask()