
import streamlit as st
import os
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.title("ChatGroq RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "file_id" not in st.session_state:
    st.session_state.file_id = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    file_id = (uploaded_file.name, uploaded_file.getvalue())

    if st.session_state.file_id != file_id:
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as temp_file:
                temp_file.write(file_id[1])
                temp_path = temp_file.name

            try:
                documents = PyPDFLoader(temp_path).load()
            finally:
                os.remove(temp_path)

            if not documents:
                st.error("No text found in this PDF.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vectorstore = FAISS.from_documents(
                chunks, embeddings
            )
            st.session_state.file_id = file_id
            st.session_state.messages = []
            st.success("Document loaded successfully!")

        except Exception as e:
            st.error(f"Could not load document: {e}")

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDF")

if question:
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please add GROQ_API_KEY to your .env file.")
    elif st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({
            "role": "user", "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        try:
            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0
            )

            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Answer using the document context below. "
                 "If the answer is not in the context, say you cannot "
                 "find it in the uploaded document.\n\n"
                 "Context:\n{context}"),
                MessagesPlaceholder("history"),
                ("human", "{question}")
            ])

            history = []
            for message in st.session_state.messages[:-1]:
                if message["role"] == "user":
                    history.append(HumanMessage(content=message["content"]))
                else:
                    history.append(AIMessage(content=message["content"]))

            chain = prompt | llm | StrOutputParser()

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = chain.invoke({
                        "context": context,
                        "history": history,
                        "question": question
                    })
                st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant", "content": answer
            })

        except Exception as e:
            st.error(f"Something went wrong: {e}")
