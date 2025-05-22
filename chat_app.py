"""Streamlit web UI for chatting with your PDF knowledge base."""

import os
from pathlib import Path

import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_chain(store_dir: Path, k: int):
    # Prefer community imports for chat model, embeddings, and vectorstores to avoid deprecation
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
    except ImportError:
        from langchain.embeddings import OpenAIEmbeddings
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    store = Chroma(
        persist_directory=str(store_dir),
        embedding_function=OpenAIEmbeddings(),
    )

    prompt = PromptTemplate(
        template=(
            "You are a helpful analyst. Answer the question using ONLY the context "
            "provided, and cite each statement with the file name and page number "
            "in square brackets, e.g. [budget_2019.pdf-p3]. If the answer is not "
            "contained in the context, say 'I don't know.'\n"
            "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
        input_variables=["question", "context"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=store.as_retriever(search_type="similarity", k=k),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def main():
    st.set_page_config(page_title="PDF-GPT", page_icon="📄")
    st.title("📄 Chat with your PDFs")

    if "OPENAI_API_KEY" not in os.environ:
        st.warning("Please set the OPENAI_API_KEY environment variable and reload.")
        st.stop()

    store_dir = Path(st.sidebar.text_input("Vector store directory", "chroma_store"))
    k = st.sidebar.slider("Top K", min_value=4, max_value=20, value=8)

    chain = _load_chain(store_dir, k)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for role, message in st.session_state.chat_history:
        align = "user" if role == "user" else "assistant"
        st.chat_message(align).write(message)

    # Input box for new question
    if user_query := st.chat_input("Ask anything about your PDFs"):
        st.session_state.chat_history.append(("user", user_query))
        st.chat_message("user").write(user_query)

        with st.spinner("Thinking..."):
            # Use invoke() to avoid deprecated __call__
            result = chain.invoke({"query": user_query})
            answer = result["result"]
            sources = {
                f"{d.metadata.get('source')}-p{d.metadata.get('page', 'NA')}": None
                for d in result["source_documents"]
            }
        st.session_state.chat_history.append(("assistant", answer))
        st.chat_message("assistant").write(answer + "\n\n**Sources:** " + ", ".join(sources.keys()))


if __name__ == "__main__":
    main()
