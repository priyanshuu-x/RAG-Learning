from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest", temperature=0.2)

loader = TextLoader("sample_document.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

decompose_prompt = PromptTemplate.from_template(
    "Break the following question into 2-3 simpler standalone sub-questions "
    "that together would help fully answer it. "
    "If the question is already simple, just return it as-is. "
    "Return ONLY the sub-questions, one per line, no numbering.\n\n"
    "Question: {question}"
)
decompose_chain = decompose_prompt | llm | StrOutputParser()

def decompose_query(question: str) -> list:
    result = decompose_chain.invoke({"question": question})
    sub_questions = [line.strip() for line in result.split("\n") if line.strip()]
    return sub_questions

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_for_subquestions(sub_questions: list) -> str:
    all_context_blocks = []

    for sub_q in sub_questions:
        docs = retriever.invoke(sub_q)
        context_block = format_docs(docs)
        all_context_blocks.append(f"[Sub-question: {sub_q}]\n{context_block}")

    return "\n\n".join(all_context_blocks)  # combined context, labeled by which sub-question it came from

answer_prompt = PromptTemplate.from_template(
    "Answer the question using ONLY the context below. The context is organized by sub-question. "
    "Combine the relevant parts into one coherent answer. If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
generation_chain = answer_prompt | llm | StrOutputParser()

def ask():
    print("Query Decomposition RAG - type 'exit' to quit\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        sub_questions = decompose_query(question)
        print(f"\nDecomposed into {len(sub_questions)} sub-question(s):")
        for sq in sub_questions:
            print(f"  - {sq}")

        combined_context = retrieve_for_subquestions(sub_questions)
        answer = generation_chain.invoke({"context": combined_context, "question": question})

        print(f"\nBot: {answer}\n")


ask()