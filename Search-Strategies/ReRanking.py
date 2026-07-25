from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever          # wraps a retriever + adds a post-processing step
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder   # the actual reranking model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest", temperature=0.2)

loader = TextLoader("sample_document.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=chunks,embedding=embeddings)

base_retriever = vectorstore.as_retriever(search_kwargs={"k":6})

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate.from_template(
    "Answer the question using ONLY the context below."
    "Context: {context}"
    "Question: {question}"
    "Answer:"
)

generation_chain = prompt | llm | StrOutputParser()

def ask():
    print("===== Reranking RAG =====")
    while True:
        question = input("You : ")
        if question.lower() == "exit":
            break

        reranked_docs = reranking_retriever.invoke(question)

        print(f" Top {len(reranked_docs)} Reranked chunks ")
        for i,doc in enumerate(reranked_docs):
            score = doc.metadata.get("relevance_score")
            print(f"\nRank {i+1} (score: {score})")
            print(doc.page_content)

        context = format_docs(reranked_docs)
        answer = generation_chain.invoke({"context": context, "question": question})

        print(f"\nBot: {answer}\n")

ask()


