# Streamlit RAG Chatbot --- 코드 주석 기반 해설 (chat.py / llm.py / config.py)

요청사항: **설명은 문서 서술이 아니라 "코드 주석"으로 코드 안에 직접
표기**했습니다.\
따라서 아래 코드를 그대로 복사하면, 코드 자체가 학습 자료가 됩니다.

------------------------------------------------------------------------

## 0) 실행 전 체크리스트

-   `.env`에 최소 아래 키가 있어야 합니다.
    -   `OPENAI_API_KEY=...`
    -   `PINECONE_API_KEY=...`
-   Pinecone에 `tax-markdown-index` 인덱스가 **이미 존재**하고, 해당
    인덱스에 문서가 **업서트**되어 있어야 합니다.
-   Streamlit 실행:
    -   `streamlit run chat.py`

------------------------------------------------------------------------

# chat.py (annotated)

``` python
import streamlit as st  # Streamlit UI 프레임워크
from dotenv import load_dotenv  # .env 파일의 환경변수를 로드하기 위한 유틸
from llm import get_ai_response  # 백엔드(RAG+LLM) 응답을 스트리밍으로 생성하는 함수

# Streamlit 앱의 기본 메타 정보(브라우저 탭 제목/아이콘)
st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

# 화면 상단 타이틀/설명 문구
st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

# .env 로드: OPENAI_API_KEY, PINECONE_API_KEY 등 실행에 필요한 키를 환경변수로 주입
load_dotenv()

# Streamlit은 기본적으로 '매 인터랙션마다 스크립트를 다시 실행'하므로,
# 대화 이력은 st.session_state에 저장해야 유지됩니다.
if "message_list" not in st.session_state:
    st.session_state.message_list = []  # [{"role": "user"|"ai", "content": "..."}]

# 저장된 과거 메시지들을 화면에 다시 렌더링 (새로고침/재실행에도 UI 유지)
for message in st.session_state.message_list:
    # Streamlit의 chat UI 컨테이너 (role에 따라 말풍선 스타일이 바뀜)
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자가 입력창에 텍스트를 입력하고 Enter를 치면 user_question에 값이 들어옵니다.
# (아무 입력도 없으면 None/False로 평가되어 아래 블록이 실행되지 않음)
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    # 1) 사용자 입력을 UI에 즉시 표시
    with st.chat_message("user"):
        st.write(user_question)

    # 2) 사용자 메시지를 세션 히스토리에 저장 (다음 rerun 때도 유지)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    # 3) 백엔드 호출 동안 로딩 스피너 표시
    with st.spinner("답변을 생성하는 중입니다"):
        # get_ai_response는 "스트리밍 iterator/generator"를 반환합니다.
        # 즉, 한 번에 문자열을 반환하는 게 아니라 토큰/청크 단위로 yield 합니다.
        ai_response_stream = get_ai_response(user_question)

        # 4) AI 말풍선 영역에 스트리밍 출력
        with st.chat_message("ai"):
            # st.write_stream은 generator를 받아서 들어오는 청크를 실시간으로 화면에 출력하고,
            # 최종적으로 화면에 출력된 전체 문자열(누적 결과)을 반환합니다.
            full_answer_text = st.write_stream(ai_response_stream)

        # 5) 최종 누적 답변을 세션에 저장 (다음 rerun에 그대로 재표시)
        st.session_state.message_list.append({"role": "ai", "content": full_answer_text})
```

------------------------------------------------------------------------

# llm.py (annotated)

``` python
# ---- 출력 파서 / 프롬프트 유틸 ----
from langchain_core.output_parsers import StrOutputParser  # LLM 출력(AIMessage 등) -> 문자열로 변환
from langchain_core.prompts import (
    ChatPromptTemplate,                 # chat prompt를 구성하는 템플릿
    MessagesPlaceholder,                # chat_history 같은 "메시지 리스트"를 프롬프트에 삽입
    FewShotChatMessagePromptTemplate,   # few-shot 예시(질문/답변)들을 chat 메시지 형태로 삽입
)

# ---- 체인 구성 요소 ----
from langchain.chains import (
    create_history_aware_retriever,  # 대화 이력을 고려해 "독립 질문"으로 재작성 후 retrieval
    create_retrieval_chain,          # retriever + 문서결합(qa) 체인을 합쳐 RAG 체인을 구성
)
from langchain.chains.combine_documents import create_stuff_documents_chain
# create_stuff_documents_chain:
#   - retrieved docs를 prompt의 {context} 자리에 "그대로(stuff)" 넣고
#   - LLM을 호출하여 답변을 생성하는 combine_docs_chain 생성

# ---- 모델/벡터스토어 ----
from langchain_openai import ChatOpenAI          # OpenAI Chat LLM 래퍼
from langchain_openai import OpenAIEmbeddings   # OpenAI 임베딩 모델 래퍼
from langchain_pinecone import PineconeVectorStore  # Pinecone 인덱스를 LangChain VectorStore로 사용

# ---- 메시지 히스토리(세션) ----
from langchain_community.chat_message_histories import ChatMessageHistory  # in-memory 메시지 저장 구현체
from langchain_core.chat_history import BaseChatMessageHistory            # 히스토리 인터페이스(추상)
from langchain_core.runnables.history import RunnableWithMessageHistory   # 체인 실행에 메시지 히스토리 자동 주입

# ---- few-shot 예시 로드 ----
from config import answer_examples  # config.py에 정의된 few-shot (input/answer) 리스트

# Streamlit 세션과 별개로, LangChain 쪽에서도 "세션별 메시지 히스토리"를 관리해야
# create_history_aware_retriever, MessagesPlaceholder("chat_history") 등이 제대로 동작합니다.
# 여기서는 간단히 파이썬 dict에 세션별 ChatMessageHistory를 저장합니다.
store = {}  # {session_id: ChatMessageHistory()}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID별 ChatMessageHistory를 반환합니다.
    - 없으면 새로 생성하여 store에 저장
    - RunnableWithMessageHistory가 이 함수를 통해 히스토리를 조회/갱신합니다.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    """Pinecone VectorStore 기반 Retriever를 생성합니다.
    - Embedding 모델: text-embedding-3-large
    - Pinecone index: tax-markdown-index (미리 구축되어 있어야 함)
    - k=4: top-4 문서 조각을 가져오도록 설정
    """
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # Pinecone에 미리 생성되어 있는 인덱스명
    index_name = "tax-markdown-index"

    # from_existing_index: 이미 존재하는 인덱스에 "접속"하는 형태 (데이터 upsert는 별도)
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )

    # retriever: .invoke(query)로 문서 조각(Document[])을 반환
    retriever = database.as_retriever(search_kwargs={"k": 4})
    return retriever


def get_llm(model: str = "gpt-4o"):
    return ChatOpenAI(model=model)


def get_history_retriever():
    """History-aware Retriever 생성.
    목적:
    - 대화형 챗봇에서 사용자가 '그거', '아까 말한', '위 내용' 같이 지시어를 쓰면
      그대로 벡터검색하면 실패할 수 있습니다.
    - 그래서 먼저 LLM으로 '독립 질문(standalone question)'으로 재작성한 후
      그 질문으로 retriever를 호출합니다.

    구성:
    - contextualize_q_prompt: (system + chat_history + latest input) -> standalone question 생성 프롬프트
    - create_history_aware_retriever: (llm, retriever, prompt)를 묶어 history-aware retriever 반환
    """
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            # 아래 placeholder에 RunnableWithMessageHistory가 session_id에 해당하는 히스토리를 자동 주입
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),  # 최신 사용자 입력
        ]
    )

    # 반환되는 history_aware_retriever는 내부적으로:
    # 1) llm으로 standalone question 생성
    # 2) retriever로 문서 검색
    # 을 수행합니다.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_dictionary_chain():
    """도메인 용어 사전(Dictionary) 기반 질문 보정 체인.
    목적:
    - 사용자 표현(사람/직장인 등)을 KB 용어(거주자 등)로 정규화해 retrieval recall/precision을 개선.
    - 이 예제에서는 아주 작은 사전 1개만 사용하지만, 실무에서는 다수 규칙/동의어/약어 테이블로 확장합니다.

    동작:
    - prompt | llm | StrOutputParser()
    - 입력 키: question
    - 출력: 보정된 질문 문자열
    """
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()

    # f-string을 사용해 사전을 프롬프트에 박아 넣었습니다.
    # (사전이 커지면: 외부 파일/DB로 분리 + 토큰 절약을 위한 구조화 권장)
    prompt = ChatPromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
그런 경우에는 질문만 리턴해주세요

사전: {dictionary}

질문: {{question}}
""")

    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain


def get_rag_chain():
    """(History-aware Retriever + Few-shot + System Prompt) 기반 RAG 체인 생성.

    큰 흐름:
    1) Few-shot(예시 Q/A) + system 지침 + chat_history + user input 으로 QA 프롬프트 구성
    2) create_stuff_documents_chain: retrieved docs를 {context}로 'stuff'하여 LLM에 전달
    3) create_retrieval_chain: history-aware retriever + combine_docs_chain 결합
    4) RunnableWithMessageHistory: session_id 기반으로 chat_history를 자동 유지/주입
    5) pick('answer'): 최종 결과 dict에서 answer만 스트림/반환
    """
    llm = get_llm()

    # ---- Few-shot 설정 ----
    # example_prompt는 "한 개 예시"의 템플릿입니다.
    # examples=answer_examples는 config.py에서 가져온 리스트이며,
    # few_shot_prompt가 이를 반복 삽입합니다.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    # ---- System prompt (행동 규칙/톤/형식) ----
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"  # create_stuff_documents_chain가 여기에 retrieved docs를 넣어줍니다.
    )

    # ---- 최종 QA Prompt ----
    # MessagesPlaceholder("chat_history")가 들어가 있으므로 multi-turn 대화가 가능해집니다.
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # ---- Retrieval + QA 결합 ----
    history_aware_retriever = get_history_retriever()

    # combine_docs_chain: (docs + prompt) -> llm -> answer
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # rag_chain: (input) -> (retriever로 docs) -> (docs+input으로 answer)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # ---- 세션 히스토리 연결 ----
    # RunnableWithMessageHistory가 session_id별로 chat_history를 자동 관리합니다.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",          # 사용자의 입력 키
        history_messages_key="chat_history", # 프롬프트 placeholder 키
        output_messages_key="answer",        # rag_chain 결과 dict에서 답변 키
    ).pick("answer")  # 최종적으로 answer 문자열만 반환/스트림

    return conversational_rag_chain


def get_ai_response(user_message: str):
    """Streamlit에서 호출하는 '스트리밍 응답' 엔트리포인트.

    여기서 하는 일:
    1) dictionary_chain으로 user_message(질문)를 KB 용어로 보정
    2) 보정된 질문을 input으로 rag_chain 실행
    3) .stream()으로 토큰/청크 단위 generator를 반환 (Streamlit이 이를 실시간 출력)

    ⚠️ 주의(키 매핑):
    - dictionary_chain은 입력 키로 {question}을 기대합니다.
    - rag_chain은 입력 키로 {input}을 기대합니다.
    - 그래서 LCEL에서 {"input": dictionary_chain}로 "dictionary 결과를 input에 바인딩"합니다.
    """
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()

    # LCEL 매핑:
    # - dictionary_chain의 출력(보정된 질문 문자열)을 rag_chain의 "input"으로 연결
    # - dictionary_chain은 실행 시 {"question": ...}를 받아야 함
    tax_chain = {"input": dictionary_chain} | rag_chain

    # stream(): generator를 반환. Streamlit의 st.write_stream이 이를 받아 실시간 표시.
    ai_response_stream = tax_chain.stream(
        {"question": user_message},
        config={
            # RunnableWithMessageHistory가 세션 히스토리를 찾는 키
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response_stream
```

------------------------------------------------------------------------

# config.py (annotated)

``` python
# few-shot 예시 목록
# - LLM에게 "답변의 형식/톤/디테일 수준"을 학습시키기 위한 샘플 Q/A들입니다.
# - FewShotChatMessagePromptTemplate에서 examples로 주입됩니다.
# - 실무에서는: 가장 대표적인 질문 유형(Top N) + 원하는 포맷을 강제하는 예시를 넣습니다.
answer_examples = [
    {
        "input": "소득은 어떻게 구분되나요?",
        "answer": """소득세법 제 4조(소득의 구분)에 따르면 소득은 아래와 같이 구분됩니다.
1. 종합소득
    - 이 법에 따라 과세되는 모든 소득에서 제2호 및 제3호에 따른 소득을 제외한 소득으로서 다음 각 목의 소득을 합산한 것
    - 가. 이자소득
    - 나. 배당소득
    - 다. 사업소득
    - 라. 근로소득
    - 마. 연금소득
    - 바. 기타소득
2. 퇴직소득
3. 양도소득
"""
    },
    {
        "input": "소득세의 과세 기간은 어떻게 되나요?",
        "answer": """소득세법 제5조(과세기간)에 따르면,
일반적인 소득세의 과세기간은 1월 1일부터 12월 31일까지 1년입니다.
하지만 거주자가 사망한 경우는 1월 1일부터 사망일까지,
거주자가 해외로 이주한 경우 1월 1일부터 출국한 날까지 입니다."""
    },
    {
        "input": "원천징수 영수증은 언제 발급받을 수 있나요?",
        "answer": """소득세법 제143조(근로소득에 대한 원천징수영수증의 발급)에 따르면,
근로소득을 지급하는 원천징수의무자는 해당 과세기간의 다음 연도 2월 말일까지
원천징수영수증을 근로소득자에게 발급해야 합니다.
다만, 해당 과세기간 중도에 퇴직한 사람에게는 퇴직한 날의 다음 달 말일까지 발급해야 하며,
일용근로자에 대하여는 근로소득의 지급일이 속하는 달의 다음 달 말일까지 발급하여야 합니다.
만약 퇴사자가 원천징수영수증을 요청한다면 지체없이 바로 발급해야 합니다."""
    },
]
```
