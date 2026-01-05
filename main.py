__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
# from dotenv import load_dotenv
# load_dotenv()

# 제목
st.title("ChatPDF")
st.write("---")

# OPENAI 키 입력받기
openai_key = st.text_input("OPENAI_API_KEY", type = "password")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type = ['pdf'])
st.write("---")

# Buy me a coffee
button(username= "skhiancgo", floating = True, width = 221)

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드된 파일 처리
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,  # 하나의 Chunk의 글자수
        chunk_overlap = 20,  # Chunk마다 겹치는 글자수 (0~300, 280~580, 560~860)
        length_function = len,  # 청크 길이를 측정하는 기준
        is_separator_regex = False,
    )

    texts = text_splitter.split_documents(pages)
    # print(texts[0], "\n\n", texts[1])

    # Embedding
    embeddings_model = OpenAIEmbeddings(
        model = "text-embedding-3-large",
        openai_api_key = openai_key
        # with the 'text-embedding-3' class
        # of models, you can specify the size
        # of the embeddings you want returned.
        # dimensions = 1024
    )

    # import chromadb
    # chromadb.ai.client.SharedSystemClient.clear_system_cache()

    # Chroma DB
    db = Chroma.from_documents(texts, embeddings_model)

    # 스트리밍 처리할 Handler 생성
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text = ""):
            self.container = container
            self.text = initial_text
        def on_llm_new_token(self, token: str, **kwarg) -> None:
            self.text += token
            self.container.markdown(self.text)

    # User Input
    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        # API 키 입력 여부 확인
        if not openai_key:
            st.error("OPENAI_API_KEY를 입력해주세요.")
            st.stop() # 이후 코드 실행을 즉시 중단합니다.
        with st.spinner("Wait for it..."):
            # Retriever
            llm = ChatOpenAI(temperature = 0, openai_api_key = openai_key)
            retriever_from_llm = MultiQueryRetriever.from_llm(  # 검색에 유리한 여러 상이한 질문으로 확장
                retriever = db.as_retriever(),
                llm = llm
            )

            # Prompt Template
            prompt = hub.pull("rlm/rag-prompt")  # 개발자들이 짜놓은 프롬프트 저장소

            # Generate
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            generate_llm = ChatOpenAI(model = "gpt-4o-mini",
                                      temperature = 0,
                                      openai_api_key = openai_key,
                                      streaming = True,
                                      callbacks = [stream_handler])
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            rag_chain = (
                {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
                | prompt
                | generate_llm
                | StrOutputParser()  # 모델의 복잡한 응답 객체에서 순수 텍스트 답변만 뽑아냄.
            )

            # Question
            result = rag_chain.invoke(question)

import os
print(os.environ.get("OPENAI_API_KEY"))
