from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest", temperature=0.3)

loader = TextLoader("sample_document.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

hyde_prompt = PromptTemplate.from_template(
    "Write a short, plausible-sounding paragraph that would answer the following question. "
    "It doesn't need to be perfectly accurate - it just needs to sound like a real answer.\n\n"
    "Question: {question}\n\n"
    "Hypothetical answer:"
)
hyde_chain = hyde_prompt | llm | StrOutputParser()

def retrieve_with_hyde(question: str, k: int = 2) -> list:
    hypothetical_answer = hyde_chain.invoke({"question": question})
    print(f"\nHypothetical answer generated:\n{hypothetical_answer}\n")

    
    hyde_vector = embeddings.embed_query(hypothetical_answer)

    docs = vectorstore.similarity_search_by_vector(hyde_vector, k=k)
    return docs

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

answer_prompt = PromptTemplate.from_template(
    "Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
generation_chain = answer_prompt | llm | StrOutputParser()

def ask():
    print("HyDE RAG - type 'exit' to quit\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        docs = retrieve_with_hyde(question)

        print(f"Retrieved {len(docs)} chunk(s) using the hypothetical answer's embedding.\n")

        context = format_docs(docs)

        answer = generation_chain.invoke({"context": context, "question": question})

        print(f"Bot: {answer}\n")

ask()