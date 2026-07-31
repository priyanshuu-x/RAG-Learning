from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest", temperature=0)

loader = TextLoader("sample_document.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

os.environ.setdefault("TAVILY_API_KEY", "")  
web_search_tool = TavilySearchResults(max_results=3)

class CragState(TypedDict):
    question: str
    documents: str
    grade: str      
    answer: str


def retrieve_node(state: CragState) -> CragState:
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(d.page_content for d in docs)
    return {"documents": context}


grade_prompt = PromptTemplate.from_template(
    "Question: {question}\n\n"
    "Retrieved text: {documents}\n\n"
    "Does the retrieved text contain enough relevant information to answer the question well? "
    "Reply with EXACTLY one word: 'relevant' or 'not_relevant'."
)
grade_chain = grade_prompt | llm | StrOutputParser()

def grade_node(state: CragState) -> CragState:
    grade = grade_chain.invoke({"question": state["question"], "documents": state["documents"]}).strip().lower()
    grade = "relevant" if "not" not in grade else "not_relevant"
    print(f"  [grade] retrieved documents judged: {grade}")
    return {"grade": grade}


def route_by_grade(state: CragState) -> Literal["generate_node", "web_search_node"]:
    return "generate_node" if state["grade"] == "relevant" else "web_search_node"

def web_search_node(state: CragState) -> CragState:
    print("  [web_search] document context was weak - searching the web instead")
    if not os.environ.get("TAVILY_API_KEY"):
        return {"documents": "Tavily API key not set - no web fallback available."}
    results = web_search_tool.invoke(state["question"])
    context = "\n\n".join(r["content"][:300] for r in results)
    return {"documents": context}

answer_prompt = PromptTemplate.from_template(
    "Answer the question using the context below.\n\n"
    "Context:\n{documents}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
answer_chain = answer_prompt | llm | StrOutputParser()

def generate_node(state: CragState) -> CragState:
    answer = answer_chain.invoke({"question": state["question"], "documents": state["documents"]})
    return {"answer": answer}


builder = StateGraph(CragState)
builder.add_node("retrieve_node", retrieve_node)
builder.add_node("grade_node", grade_node)
builder.add_node("web_search_node", web_search_node)
builder.add_node("generate_node", generate_node)

builder.add_edge(START, "retrieve_node")
builder.add_edge("retrieve_node", "grade_node")
builder.add_conditional_edges("grade_node", route_by_grade)
builder.add_edge("web_search_node", "generate_node")   # after web fallback, still generate normally
builder.add_edge("generate_node", END)

graph = builder.compile()

def ask():
    print("Corrective RAG (type 'exit' to quit)\n")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        result = graph.invoke({"question": question, "documents": "", "grade": "", "answer": ""})
        print(f"\nBot: {result['answer']}\n")

if __name__ == "__main__":
    ask()