# 1. SQLite 패치 (Streamlit Cloud 배포용)
# 이 코드는 반드시 다른 임포트보다 최상단에 있어야 합니다.
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # 로컬(Windows) 환경 등 pysqlite3가 없는 경우 패스합니다.
    pass

import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

# LangChain 관련 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler

# 제목
st.title("ChatPDF")
st.write("---")

# OPENAI 키 입력받기 (공백 제거 기능 추가)
openai_key = st.text_input("OPENAI_API_KEY", type="password").strip()

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

# Buy me a coffee
button(username="skhiancgo", floating=True, width=221)

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 스트리밍 핸들러 정의
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwarg) -> None:
        self.text += token
        self.container.markdown(self.text)

# --- 메인 로직 시작 ---

# 1. API 키가 없으면 경고 문구만 띄우고 진행하지 않음 (에러 방지 핵심)
if not openai_key:
    st.info("👋 API 키를 입력해주시면 PDF 분석을 시작할 수 있습니다.")
    st.stop()

# 2. 파일이 업로드 되었을 때만 실행
if uploaded_file is not None:
    with st.spinner("PDF 문서를 분석하고 있습니다... 잠시만 기다려주세요."):
        # PDF 변환
        pages = pdf_to_document(uploaded_file)

        # Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(pages)

        # Embedding & DB Creation
        # API 키가 확실히 있을 때만 생성
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=openai_key
        )
        
        # Chroma DB 생성
        db = Chroma.from_documents(texts, embeddings_model)

    # 3. 사용자 질문 입력 및 처리
    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        if not question:
            st.warning("질문을 입력해주세요.")
            st.stop()

        with st.spinner("답변을 생성하고 있습니다..."):
            # Retriever 설정
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=openai_key
            )
            
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=db.as_retriever(),
                llm=llm
            )

            # Prompt & Chain
            prompt = hub.pull("rlm/rag-prompt")

            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            
            generate_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=openai_key,
                streaming=True,
                callbacks=[stream_handler]
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
                | prompt
                | generate_llm
                | StrOutputParser()
            )

            # 실행
            try:
                result = rag_chain.invoke(question)
            except Exception as e:
                st.error(f"에러가 발생했습니다: {e}")

