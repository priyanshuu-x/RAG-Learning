from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest", temperature=0)

loader = TextLoader("sample_document.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

@tool
def retrieve_document_context(query: str) -> str:
    """Search the document for information relevant to a query. Can be called multiple
    times with different, more specific queries if the first result isn't sufficient."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join(doc.page_content for doc in docs)

tools = [retrieve_document_context]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

SYSTEM_PROMPT = SystemMessage(content=(
    "You answer questions about AI, using the retrieve_document_context tool when you "
    "need information from the document. You do NOT need to retrieve for questions you "
    "can answer directly (e.g. simple math, general knowledge outside the document's scope - "
    "in that case just say the document doesn't cover it). If retrieved context isn't enough "
    "to answer well, you may call the tool again with a more specific query."
))

def chat_node(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

builder = StateGraph(AgentState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

graph = builder.compile()

def ask():
    print("Agentic RAG (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        result = graph.invoke({"messages": [HumanMessage(content=question)]})

        for msg in result["messages"]:
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    print(f"  [retrieved with query]: {call['args'].get('query')}")

        print(f"\nBot: {result['messages'][-1].content}\n")

if __name__ == "__main__":
    ask()