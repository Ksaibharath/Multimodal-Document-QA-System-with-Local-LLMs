# Multimodal PDF RAG Chatbot using OCR, ChromaDB, and Ollama

## Overview

This project is a **Multimodal PDF Question Answering System** built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload PDF documents and ask questions in natural language. The system extracts content from the document, stores semantic embeddings in a vector database, retrieves the most relevant chunks, and generates answers locally using **Ollama**.

The application also includes **OCR support** for scanned or image-based PDFs, **page image rendering** for visual context, **page-level citations**, and **chat history** for a more interactive user experience.

This project was built as a hands-on implementation of:
- document intelligence
- retrieval-augmented generation
- local LLM integration
- OCR pipelines
- multimodal document understanding foundations

---

## Key Features

- Upload PDF reports, documents, or assignments
- Extract text from standard digital PDFs
- Apply OCR when the PDF page has no extractable text
- Split text into chunks for semantic retrieval
- Generate embeddings using Sentence Transformers
- Store and search embeddings in ChromaDB
- Render PDF pages as images for visual reference
- Ask natural language questions about the uploaded document
- Retrieve top relevant chunks with page metadata
- Generate final answers locally using Ollama
- Show best matching page image for visual context
- Display page citations for transparency
- Maintain chat history across questions
- Filter search results to the currently uploaded file
- Clear vector database and chat history from the UI

---

## Why This Project?

Reading long PDF reports manually is time-consuming. Traditional keyword search is often not enough because users may ask questions in natural language such as:

- “What is the title of the document?”
- “Summarize this report.”
- “What is the conclusion?”
- “What does page 1 discuss?”
- “Which topics are covered in the document?”

This project solves that problem by combining:
- semantic retrieval
- OCR-based text recovery
- local LLM generation
- page-level visual context

It acts like a **local ChatPDF-style assistant**.

---

## Problem Statement

PDFs often contain:
- large amounts of text
- scanned pages
- difficult layouts
- figures and diagrams
- content that is hard to search manually

A normal LLM alone cannot answer accurately unless relevant context is provided. Therefore, this project uses **RAG (Retrieval-Augmented Generation)** to first retrieve relevant content from the PDF and then generate an answer based on that retrieved context.

---

## What Makes It “Multimodal”?

This project is called multimodal because it uses more than one type of document representation:

1. **Text modality**
   - extracted PDF text
   - OCR text from non-readable pages

2. **Visual modality**
   - rendered images of PDF pages
   - best matching page preview shown to the user

At the current stage, the system uses images primarily for page preview and visual context. It is designed so that it can later be extended to support:
- chart understanding
- figure understanding
- image captioning
- vision-language models

---

## System Architecture

### 1. Document Ingestion Pipeline

When a PDF is uploaded:

1. The PDF is read page by page using **PyMuPDF**
2. Text is extracted from each page
3. If a page contains no extractable text, **EasyOCR** is used
4. Extracted text is split into smaller chunks
5. Embeddings are generated for each chunk using **Sentence Transformers**
6. Chunks and metadata are stored in **ChromaDB**
7. Each PDF page is rendered as an image and stored locally

### 2. Query Pipeline

When the user asks a question:

1. The question is converted into an embedding
2. ChromaDB retrieves the most relevant chunks
3. Retrieval is filtered to the currently uploaded PDF
4. Retrieved chunks are passed to the local **Ollama** model
5. Ollama generates a final answer based only on the retrieved context
6. The UI shows:
   - final answer
   - document title guess
   - citation pages
   - best matching page preview
   - source chunks

---

## Tech Stack

### Programming Language
- **Python**

### Frontend
- **Streamlit**

### PDF Processing
- **PyMuPDF (fitz)**

### OCR
- **EasyOCR**

### Embeddings
- **Sentence Transformers**
- Model used: `all-MiniLM-L6-v2`

### Vector Database
- **ChromaDB**

### Local LLM Inference
- **Ollama**
- Recommended local model: `phi3:latest`

### Supporting Libraries
- Pillow
- NumPy
- Requests

---

## Folder Structure

```bash
multimodal-rag/
│
├── app.py                  # Streamlit application
├── utils.py                # Core processing, retrieval, OCR, Ollama functions
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── uploaded_files/         # Uploaded PDFs
├── page_images/            # Rendered PDF page images
├── chroma_db/              # Persistent Chroma database
└── venv/                   # Virtual environment (optional, not pushed to GitHub)