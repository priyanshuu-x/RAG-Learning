import fitz
import pymupdf
from PIL import Image
import torch
import numpy as np
import io 
import base64

from transformers import CLIPProcessor , CLIPModel
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def embed_image(pil_image: Image.Image) -> np.ndarray:
    """Turn a PIL image into a CLIP vector."""
    inputs = clip_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():  # no_grad -> we're not training, saves memory/computation
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)  # normalize to unit length
        return features.squeeze().numpy()

def embed_text(text: str) -> np.ndarray:
    """Turn a string into a CLIP vector (same vector space as embed_image)."""
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()



pdf_path = "multimodal_sample.pdf"  # put your own PDF here
pdf = fitz.open(pdf_path)
 
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
 
all_docs = []           # will hold Document objects for BOTH text chunks and images
all_embeddings = []      # matching CLIP vector for each entry in all_docs
image_data_store = {}    # image_id -> base64 string, kept separately (too big to put in metadata)
 
for page_num, page in enumerate(pdf):
    
    text = page.get_text()
    if text.strip():
        temp_doc = Document(page_content=text, metadata={"page": page_num, "type": "text"})
        for chunk in splitter.split_documents([temp_doc]):
            all_docs.append(chunk)
            all_embeddings.append(embed_text(chunk.page_content))
 
    
    for img_index, img in enumerate(page.get_images(full=True)):
        try:
            xref = img[0]
            image_bytes = pdf.extract_image(xref)["image"]
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
 
            image_id = f"page_{page_num}_img_{img_index}"
 
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            image_data_store[image_id] = base64.b64encode(buffer.getvalue()).decode()
 
            all_embeddings.append(embed_image(pil_image))
            all_docs.append(Document(
                page_content=f"[Image: {image_id}]",
                metadata={"page": page_num, "type": "image", "image_id": image_id}
            ))
        except Exception as e:
            print(f"Skipped image {img_index} on page {page_num}: {e}")
 
pdf.close()
print(f"Extracted {len(all_docs)} total chunk(s) (text + images) from the PDF.\n")

vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, all_embeddings)],
    embedding=None,  # no embedding model needed - vectors are already computed
    metadatas=[doc.metadata for doc in all_docs])

llm = ChatOllama(model="llama3.2-vision", temperature=0.2)

def retrieve_multimodal(query: str, k: int = 4) -> list:
    query_embedding = embed_text(query)  # query embedded with CLIP, same space as text AND images
    return vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)


def build_multimodal_message(query: str, retrieved_docs: list) -> HumanMessage:
    text_docs = [d for d in retrieved_docs if d.metadata.get("type") == "text"]
    image_docs = [d for d in retrieved_docs if d.metadata.get("type") == "image"]
 
    content = [{"type": "text", "text": f"Question: {query}\n\nContext:\n"}]
 
    if text_docs:
        text_context = "\n\n".join(f"[Page {d.metadata['page']}]: {d.page_content}" for d in text_docs)
        content.append({"type": "text", "text": f"Text excerpts:\n{text_context}\n"})
 
    for d in image_docs:
        image_id = d.metadata.get("image_id")
        if image_id in image_data_store:
            content.append({"type": "text", "text": f"\n[Image from page {d.metadata['page']}]:\n"})
            # image_url with a base64 data URI - this is how images get attached to a chat message
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[image_id]}"}
            })
 
    content.append({"type": "text", "text": "\n\nAnswer the question using the text and images above."})
    return HumanMessage(content=content)



def multimodal_rag(query: str) -> str:
    docs = retrieve_multimodal(query)
 
    print(f"Retrieved {len(docs)} chunk(s):")
    for d in docs:
        kind = d.metadata.get("type")
        page = d.metadata.get("page")
        if kind == "text":
            preview = d.page_content[:80] + "..." if len(d.page_content) > 80 else d.page_content
            print(f"  - [text]  page {page}: {preview}")
        else:
            print(f"  - [image] page {page}")
    print()
 
    message = build_multimodal_message(query, docs)
    response = llm.invoke([message])
    return response.content
 
if __name__ == "__main__":
    print("Multimodal RAG (type 'exit' to quit)\n")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        answer = multimodal_rag(question)
        print(f"Bot: {answer}\n")

