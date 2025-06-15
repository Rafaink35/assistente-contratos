import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
st.set_page_config(page_title="Assistente de Contratos", page_icon="📄")
st.title("📄 Assistente de Contratos (OpenRouter + FAISS)")
st.caption("Gratuito usando modelos open source via OpenRouter (sem limite de tokens da OpenAI).")

if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.memory = None

with st.form("upload_form"):
    st.subheader("📥 Upload e configurações")

    api_key_input = st.text_input("🔑 OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
    uploaded_file = st.file_uploader("📤 Faça upload do contrato (PDF ou TXT)", type=["pdf", "txt"])
    top_k = st.slider("🔎 Quantidade de trechos analisados", 2, 10, 4)
    temperature = st.slider("🔥 Temperatura da resposta", 0.0, 1.0, 0.1)

    submitted = st.form_submit_button("🔄 Processar documento")

def build_chain(file, api_key: str):
    with st.spinner("🔧 Processando documento…"):
        suffix = Path(file.name).suffix

        if file.size > 50_000_000:
            st.warning("⚠️ O arquivo é muito grande. Tente um menor que 50MB.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embeddings)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chat = ChatOpenAI(
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model_name="mistralai/mistral-7b-instruct",
            temperature=temperature
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=vectordb.as_retriever(search_kwargs={"k": top_k}),
            memory=memory,
            return_source_documents=True,
        )
    return chain, memory

if submitted:
    if not uploaded_file or not api_key_input:
        st.warning("⚠️ Envie um contrato e sua chave da OpenRouter.")
    else:
        st.session_state.chain, st.session_state.memory = build_chain(uploaded_file, api_key_input)
        st.success("✅ Documento carregado! Agora você pode fazer perguntas abaixo.")

query = st.chat_input("Digite sua pergunta sobre o contrato...")

if query and st.session_state.chain:
    with st.spinner("Consultando o modelo…"):
        result = st.session_state.chain(query)
        answer = result["answer"]
        sources = result["source_documents"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    if sources:
        with st.expander("🔍 Fontes citadas no contrato"):
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**{i}. Página {page}**")
                st.write(doc.page_content[:500] + "…")

if st.session_state.memory:
    with st.expander("🗂️ Histórico da Conversa"):
        for msg in st.session_state.memory.chat_memory.messages:
            role = "👤 Você" if msg.type == "human" else "🤖 Assistente"
            st.markdown(f"**{role}:** {msg.content}")
