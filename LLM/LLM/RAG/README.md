# Base RAG 구성 --- LangChain + Chroma + OpenAI Embeddings (실무 완전 이해 버전)

이 문서는 이후 모든 RAG 실습의 기준이 되는 **Base Architecture
문서**입니다.\
단순 코드 설명이 아니라, **실무에서 RAG 시스템을 직접 설계하고 구현할 수
있는 수준까지 이해하는 것을 목표**로 합니다.

------------------------------------------------------------------------

# 실습 주제

## 이 실습의 목적

이 실습의 핵심 목적은 다음입니다:

1.  외부 문서를 Vector Database에 저장하는 방법 이해
2.  사용자 질문과 유사한 문서를 검색하는 방법 이해
3.  검색된 문서를 기반으로 LLM이 답변을 생성하는 구조 이해
4.  LangChain을 활용하여 전체 흐름을 구성하는 방법 이해

이 구조는 모든 RAG 시스템의 기반입니다.

------------------------------------------------------------------------

## 어떤 문제를 해결하기 위한 것인가

일반 LLM의 한계:

예시:

질문: "우리 회사 세금 계산 정책은?"

일반 LLM:

→ 모름 (학습되지 않은 데이터)

RAG:

→ 회사 문서 검색 → 검색된 문서 기반 답변 생성

즉,

LLM + External Knowledge = RAG

------------------------------------------------------------------------

## 일반 LLM vs RAG 차이

일반 LLM:

Input: "연봉 5000 소득세 얼마?"

LLM:

→ 학습된 일반 지식 기반 답변

문제:

-   최신 정보 아님
-   회사 정책 반영 불가
-   hallucination 발생 가능

------------------------------------------------------------------------

RAG:

Input: "연봉 5000 소득세 얼마?"

동작:

1.  Vector DB에서 관련 문서 검색
2.  검색된 문서를 Prompt에 포함
3.  LLM이 문서 기반 답변 생성

결과:

-   정확도 상승
-   hallucination 감소
-   최신 데이터 사용 가능

------------------------------------------------------------------------

# 전체 아키텍처 (핵심)

    User Question 
    ↓ 
    Embedding Model (질문 → vector 변환) 
    ↓ 
    Vector Store(Chroma) 
    ↓ 
    Similarity Search 
    ↓ 
    Relevant Documents 
    ↓ 
    Prompt Template 
    ↓
    LLM 
    ↓ 
    Final Answer

------------------------------------------------------------------------

# 전체 코드 + 완전 상세 설명

------------------------------------------------------------------------

# Step 1 --- 패키지 설치

``` python
%pip install python-dotenv langchain langchain-openai langchain-community langchain-text-splitters docx2txt langchain-chroma
```

각 패키지 역할:

    패키지                     역할
    -------------------------- ---------------------------
    langchain                  전체 framework
    langchain-openai           OpenAI model 연결
    langchain-community        loader 등 community tools
    langchain-text-splitters   문서 분할
    docx2txt                   docx 파일 읽기
    langchain-chroma           Vector DB

------------------------------------------------------------------------

# Step 2 --- Document Loader

``` python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader('./tax.docx')
```

역할:

docx 파일 → LangChain Document 객체 변환

Document 구조:

``` python
Document(
    page_content="텍스트 내용",
    metadata={}
)
```

------------------------------------------------------------------------

# Step 3 --- Text Splitter

``` python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, # 각 chunk 길이
    chunk_overlap=200, # 겹치게 할 길이
)
```

왜 Split이 필요한가?

LLM은 입력 token 제한이 있음

예:

GPT-4 입력 제한 존재

따라서:

큰 문서 → 작은 chunk로 분할 필요

------------------------------------------------------------------------

Split 실행:

``` python
document_list = loader.load_and_split(text_splitter=text_splitter)
```

결과:

\[ Document(chunk1), Document(chunk2), Document(chunk3)\]

------------------------------------------------------------------------

# Step 4 --- Embedding 생성

``` python
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large')
```

Embedding이란?

텍스트 → 숫자 vector 변환

예:

"소득세 계산"

→

\[0.123, -0.532, 0.998, ...\]

왜 필요한가?

컴퓨터는 의미를 숫자로 비교함

------------------------------------------------------------------------

Vector 유사도 계산 방식:

Cosine similarity:

유사한 문장 → 가까운 vector

------------------------------------------------------------------------

# Step 5 --- Vector Database 생성

``` python
from langchain_chroma import Chroma

database = Chroma(
    collection_name='chroma-tax',
    persist_directory="./chroma", # 임베딩결과 로컬저장
    embedding_function=embedding
)
```

------------------------------------------------------------------------

# Step 6 --- Retrieval

``` python
query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'

retrieved_docs = database.similarity_search(query, k=3)
```

동작:

1.  query → vector 변환
2.  DB에서 유사한 vector 검색
3.  top k 반환

k=3 의미:

가장 유사한 문서 3개

------------------------------------------------------------------------

# Step 7 --- Prompt Template

``` python
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")
```

rag-prompt 구조:

Context: {retrieved_docs}

Question: {query}

------------------------------------------------------------------------

# Step 8 --- LLM 생성

``` python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o')
```
------------------------------------------------------------------------

# Step 9 --- Chain 생성

``` python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=database.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
```

Chain 역할:

Retriever + Prompt + LLM 연결

------------------------------------------------------------------------

# Step 10 --- 실행

``` python
ai_message = qa_chain.invoke({"query": query})
```

전체 실행 흐름:

query → embedding → vector search → docs 반환 → prompt 생성 → llm 실행 →
답변 생성

------------------------------------------------------------------------

# 내부 동작 (핵심)

실제 내부 흐름:

query

→ embedding

→ similarity search

→ docs

→ prompt 구성

→ llm

→ answer

------------------------------------------------------------------------

# Production에서 이 구조가 중요한 이유

이 구조가 모든 RAG 시스템의 기본입니다.

ChatGPT Enterprise Perplexity 사내 AI 시스템

모두 동일 구조

------------------------------------------------------------------------

# 이후 실습에서 변경될 요소

Embedding 변경

Vector DB 변경

Chain 변경

Retrieval 전략 변경

------------------------------------------------------------------------

# 결론

이 구조를 완전히 이해하면:

-   어떤 Vector DB든 사용 가능
-   어떤 Embedding Model든 사용 가능
-   어떤 LLM이든 연결 가능

즉,

완전한 RAG 시스템 구현 가능
