import os
import streamlit as st
from utils import (
    process_pdf,
    retrieve_relevant_chunks,
    get_document_title,
    generate_simple_summary,
    clear_vector_database,
    generate_answer_with_ollama,
    get_citation_pages
)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Multimodal PDF RAG Chatbot", layout="wide")

st.title("Multimodal PDF RAG Chatbot")
st.caption("Local PDF Question Answering with OCR, Retrieval, Page Images, and Ollama")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header("Controls")

if st.sidebar.button("Clear Vector Database"):
    clear_vector_database()
    st.sidebar.success("Vector database cleared.")

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared.")

ollama_model = st.sidebar.selectbox(
    "Choose Ollama Model",
    ["phi3:latest"],
    index=0
)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Uploaded file: {uploaded_file.name}")

    if st.button("Process PDF"):
        with st.spinner("Processing PDF... please wait"):
            num_chunks, num_page_images, page_images = process_pdf(file_path, uploaded_file.name)

        st.success("PDF processed successfully.")
        col1, col2 = st.columns(2)
        col1.metric("Stored Text Chunks", num_chunks)
        col2.metric("Rendered Page Images", num_page_images)

        if page_images:
            st.subheader("PDF Page Preview")
            for img in page_images[:3]:
                with st.expander(f"Page {img['page']} | Source: {img['source']}"):
                    st.image(img["image_path"], width=500)

st.divider()

question = st.text_input("Ask a question about the uploaded PDF:")

if st.button("Get Answer"):
    if not uploaded_file:
        st.warning("Please upload and process a PDF first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant content..."):
            retrieved_chunks = retrieve_relevant_chunks(
                question,
                top_k=3,
                source_filter=uploaded_file.name
            )

        if not retrieved_chunks:
            st.warning("No relevant content found for this PDF.")
        else:
            with st.spinner(f"Generating answer using Ollama ({ollama_model})..."):
                final_answer = generate_answer_with_ollama(
                    question,
                    retrieved_chunks,
                    model_name=ollama_model
                )

            title = get_document_title(retrieved_chunks)
            summary = generate_simple_summary(retrieved_chunks)
            citation_pages = get_citation_pages(retrieved_chunks)
            best_page = retrieved_chunks[0]["page"]

            st.session_state.chat_history.append({
                "question": question,
                "title": title,
                "summary": summary,
                "final_answer": final_answer,
                "results": retrieved_chunks,
                "citations": citation_pages,
                "best_page": best_page,
                "source": uploaded_file.name
            })

if st.session_state.chat_history:
    st.subheader("Chat History")

    for idx, chat in enumerate(reversed(st.session_state.chat_history), start=1):
        st.markdown(f"## Question {idx}")
        st.markdown(f"**You asked:** {chat['question']}")

        st.markdown("### Final Answer")
        st.success(chat["final_answer"])

        st.markdown("### Source Information")
        col1, col2, col3 = st.columns(3)
        col1.info(f"**Document:** {chat['source']}")
        col2.info(f"**Best Page:** {chat['best_page']}")
        col3.info(f"**Citations:** {', '.join(map(str, chat['citations']))}")

        st.markdown("### Likely Document Title")
        st.info(chat["title"])

        st.markdown("### Retrieved Summary")
        st.write(chat["summary"])

        best_page_image = os.path.join("page_images", f"{chat['source']}_page_{chat['best_page']}.png")
        if os.path.exists(best_page_image):
            st.markdown("### Best Matching Page Preview")
            st.image(best_page_image, width=600)

        st.markdown("### Source Chunks")
        for i, chunk in enumerate(chat["results"], start=1):
            page_image_path = os.path.join("page_images", f"{chunk['source']}_page_{chunk['page']}.png")

            with st.expander(f"Chunk {i} | Page {chunk['page']} | Source: {chunk['source']}"):
                st.write(chunk["text"])

                if os.path.exists(page_image_path):
                    st.markdown("**Related Page Image:**")
                    st.image(page_image_path, width=500)

        st.markdown("---")