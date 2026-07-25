from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest", temperature=0.5)  # a little creativity helps generate varied phrasings

loader = TextLoader("sample_document.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# -------------------------------
# STEP 2: Ask the LLM to generate reworded versions of the user's query
# -------------------------------
expansion_prompt = PromptTemplate.from_template(
    "Generate 3 different ways to phrase the following question, "
    "using different words but keeping the same meaning. "
    "Return ONLY the 3 questions, one per line, no numbering.\n\n"
    "Question: {question}"
)
expansion_chain = expansion_prompt | llm | StrOutputParser()

def expand_query(question: str) -> list:
    result = expansion_chain.invoke({"question": question})
    variants = [line.strip() for line in result.split("\n") if line.strip()]  # split into a clean list
    return variants

# -------------------------------
# STEP 3: Retrieve using the ORIGINAL query + all expanded variants, then merge results
# -------------------------------
def retrieve_with_expansion(question: str) -> list:
    all_queries = [question] + expand_query(question)  # original query stays in the mix too

    print(f"\nSearching with {len(all_queries)} query variant(s):")
    for q in all_queries:
        print(f"  - {q}")

    seen_content = set()  # tracks chunk text we've already added, to avoid duplicates
    merged_docs = []

    for q in all_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                merged_docs.append(doc)

    return merged_docs

# -------------------------------
# STEP 4: Generation chain
# -------------------------------
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
    print("Query Expansion RAG - type 'exit' to quit\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        merged_docs = retrieve_with_expansion(question)
        print(f"\nRetrieved {len(merged_docs)} unique chunk(s) across all variants.\n")

        context = format_docs(merged_docs)
        answer = generation_chain.invoke({"context": context, "question": question})

        print(f"Bot: {answer}\n")

ask()