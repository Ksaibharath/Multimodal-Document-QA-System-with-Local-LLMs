import os
import fitz
import chromadb
import easyocr
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import requests

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

ocr_reader = None


def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        print("Loading EasyOCR model...")
        ocr_reader = easyocr.Reader(['en'], gpu=False)
    return ocr_reader


def extract_text_with_ocr(page):
    reader = get_ocr_reader()

    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)

    results = reader.readtext(img_np, detail=0)
    text = " ".join(results)

    return text


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        if not text:
            print(f"Page {page_num}: No normal text found. Using OCR...")
            text = extract_text_with_ocr(page)

        if text.strip():
            pages.append({
                "page": page_num,
                "text": text
            })

    return pages


def render_pdf_pages_as_images(pdf_path, filename):
    doc = fitz.open(pdf_path)
    output_dir = "page_images"
    os.makedirs(output_dir, exist_ok=True)

    page_images = []

    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        image_name = f"{filename}_page_{page_num}.png"
        image_path = os.path.join(output_dir, image_name)
        pix.save(image_path)

        page_images.append({
            "page": page_num,
            "image_path": image_path,
            "source": filename
        })

    return page_images


def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def process_pdf(pdf_path, filename):
    pages = extract_text_from_pdf(pdf_path)
    page_images = render_pdf_pages_as_images(pdf_path, filename)

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_page_{page_num}_text_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": filename,
                "page": page_num,
                "type": "text"
            })

    if all_chunks:
        existing = collection.get(ids=all_ids)
        existing_ids = set(existing["ids"]) if existing["ids"] else set()

        new_chunks = []
        new_ids = []
        new_metas = []

        for chunk, chunk_id, meta in zip(all_chunks, all_ids, all_metadatas):
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)
                new_metas.append(meta)

        if new_chunks:
            embeddings = embedding_model.encode(new_chunks).tolist()

            collection.add(
                ids=new_ids,
                embeddings=embeddings,
                documents=new_chunks,
                metadatas=new_metas
            )

    return len(all_chunks), len(page_images), page_images


def retrieve_relevant_chunks(query, top_k=3, source_filter=None):
    query_embedding = embedding_model.encode([query]).tolist()[0]

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k
    }

    if source_filter:
        query_args["where"] = {"source": source_filter}

    results = collection.query(**query_args)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved = []
    for doc, meta in zip(documents, metadatas):
        retrieved.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "type": meta.get("type", "text")
        })

    return retrieved


def clear_vector_database():
    global collection
    chroma_client.delete_collection(name="pdf_chunks")
    collection = chroma_client.get_or_create_collection(name="pdf_chunks")


def get_document_title(retrieved_chunks):
    if not retrieved_chunks:
        return "Title not found"

    first_text = retrieved_chunks[0]["text"]
    return first_text[:120]


def generate_simple_summary(retrieved_chunks):
    if not retrieved_chunks:
        return "No summary available."

    combined = " ".join([chunk["text"] for chunk in retrieved_chunks[:2]])
    words = combined.split()

    short_summary = " ".join(words[:60])
    return short_summary + "..."


def get_citation_pages(retrieved_chunks):
    pages = sorted(list(set(chunk["page"] for chunk in retrieved_chunks)))
    return pages


def generate_answer_with_ollama(question, retrieved_chunks, model_name="phi3:latest"):
    short_context = "\n\n".join([
        f"Page {chunk['page']}: {chunk['text'][:300]}"
        for chunk in retrieved_chunks[:3]
    ])

    prompt = f"""
Answer the question using only this context.

If the answer is not in the context, say:
I could not find the answer in the uploaded document.

Also mention the relevant page number(s) if possible.

Context:
{short_context}

Question:
{question}

Give a short answer in 2-4 lines.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response received from Ollama.")

    except Exception as e:
        return f"Ollama error: {str(e)}"