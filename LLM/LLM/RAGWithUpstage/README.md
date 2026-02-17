# 실습 01 --- Embedding Model 변경 (OpenAI → Upstage Embeddings)

이 문서는 Base RAG 구조에서 **Embedding Model과 LLM Provider를
OpenAI에서 Upstage로 변경한 부분만 설명**합니다.

Base 구조 설명은 아래 문서를 기준으로 합니다:

00_base_rag_langchain_chroma_FULL.md

------------------------------------------------------------------------

# 변경 목적

기존 구조:

Embedding Model: OpenAIEmbeddings (text-embedding-3-large)

LLM: ChatOpenAI (gpt-4o)

변경 구조:

Embedding Model: UpstageEmbeddings (solar-embedding-1-large)

LLM: ChatUpstage (solar LLM)

------------------------------------------------------------------------

# 왜 Embedding Model을 변경하는가

Embedding Model은 RAG 성능에 직접적인 영향을 줍니다.

역할:

텍스트 의미 → vector 변환

Embedding 품질이 좋을수록:

-   검색 정확도 증가
-   hallucination 감소
-   retrieval relevance 증가

------------------------------------------------------------------------

Upstage Embedding 특징:

모델: solar-embedding-1-large

특징:

-   한국어 성능 매우 우수
-   OpenAI embedding 대비 한국어 semantic search 성능 우수
-   한국어 RAG 시스템에 매우 적합

------------------------------------------------------------------------

# 변경된 코드

## 변경 전 (Base)

``` python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
```

------------------------------------------------------------------------

## 변경 후 (Upstage)

``` python
from langchain_upstage import UpstageEmbeddings

embedding = UpstageEmbeddings(
    model="solar-embedding-1-large"
)
```

------------------------------------------------------------------------

# 내부 동작 차이

동작 방식은 동일합니다.

Text:

"소득세 계산"

→ Embedding Model

→ Vector:

\[0.123, -0.532, 0.999, ...\]

차이점:

Vector 품질

Upstage:

-   한국어 semantic 표현 우수

OpenAI:

-   영어 중심 최적화

------------------------------------------------------------------------

# Vector DB 동작은 동일

``` python
database = Chroma.from_documents(
    documents=document_list,
    embedding=embedding,
    collection_name='chroma-tax',
    persist_directory="./chroma"
)
```

Chroma는 embedding model과 무관하게 동일하게 동작합니다.

------------------------------------------------------------------------

# LLM 변경

기존:

``` python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
```

변경:

``` python
from langchain_upstage import ChatUpstage

llm = ChatUpstage()
```

------------------------------------------------------------------------

# ChatUpstage 특징

Provider:

Upstage Solar LLM

특징:

-   한국어 성능 매우 우수
-   한국 기업 환경에 적합
-   OpenAI 대비 비용 효율적

------------------------------------------------------------------------

# Retrieval 및 Chain 구성은 동일

``` python
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=database.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
```

------------------------------------------------------------------------

# 실행 코드

``` python
ai_message = qa_chain.invoke({"query": query})
```

------------------------------------------------------------------------

# 아키텍처 비교

Before:

User ↓ OpenAI Embedding ↓ Chroma ↓ ChatOpenAI

After:

User ↓ Upstage Embedding ↓ Chroma ↓ ChatUpstage

------------------------------------------------------------------------

# 실무에서 Upstage Embedding을 사용하는 경우

사용 권장:

-   한국어 서비스
-   한국 문서 기반 RAG
-   비용 최적화 필요 시

OpenAI 권장:

-   영어 중심 서비스
-   글로벌 서비스

------------------------------------------------------------------------

# 성능 영향

한국어 기준:

Upstage Embedding \> OpenAI Embedding

Retrieval Accuracy 증가

------------------------------------------------------------------------

# 결론

변경된 부분은 단 2개:

Embedding Model

OpenAI → Upstage

LLM Provider

ChatOpenAI → ChatUpstage

나머지 RAG 구조는 완전히 동일합니다.
