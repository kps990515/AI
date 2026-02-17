# 실습 04 --- Vector DB + Embedding/LLM 변경 (Pinecone + Upstage)

이 문서는 이전 실습들을 기준으로 **추가로 변경된 부분만 설명**합니다.

기준 문서:

-   Base RAG 구조: `00_base_rag_langchain_chroma_FULL.md`
-   Embedding 변경(OpenAI → Upstage): `01_rag_upstage_embedding.md`
-   Vector DB 변경(Chroma → Pinecone):
    `03_vector_db_chroma_to_pinecone.md`

이번 실습은 위 두 변경을 **동시에 적용한 구조**입니다.

------------------------------------------------------------------------

# 변경 요약

  구성요소    이전               현재
  ----------- ------------------ -------------------
  Vector DB   Pinecone           Pinecone (동일)
  Embedding   OpenAIEmbeddings   UpstageEmbeddings
  LLM         ChatOpenAI         ChatUpstage

즉,

Vector DB는 Pinecone 유지\
Embedding + LLM만 Upstage로 변경

------------------------------------------------------------------------

# 변경 목적

이 구조는 다음 목적을 위한 **Production-ready 한국어 RAG 구조**입니다:

-   Vector DB: Pinecone (운영 환경용)
-   Embedding: Upstage (한국어 semantic search 최적화)
-   LLM: Upstage Solar (한국어 생성 성능 최적화)

------------------------------------------------------------------------

# 변경된 코드 --- Embedding

Before (OpenAI + Pinecone):

``` python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
```

After (Upstage + Pinecone):

``` python
from langchain_upstage import UpstageEmbeddings

embedding = UpstageEmbeddings(
    model="solar-embedding-1-large"
)
```

------------------------------------------------------------------------

# 내부 동작 차이 (Embedding)

동작은 동일:

Text

→ Embedding Model

→ Vector

→ Pinecone 저장

차이:

Vector 품질

Upstage:

-   한국어 semantic 표현 매우 우수
-   한국어 retrieval accuracy 향상

------------------------------------------------------------------------

# 변경된 코드 --- Pinecone VectorStore 생성

``` python
from langchain_pinecone import PineconeVectorStore

index_name = 'tax-upstage-index'

database = PineconeVectorStore.from_documents(
    document_list,
    embedding,
    index_name=index_name
)
```

핵심:

Embedding만 Upstage로 변경되었고 Pinecone 구조는 동일

------------------------------------------------------------------------

# Pinecone 내부 저장 구조

각 document chunk:

저장 형태:

Vector: \[0.123, -0.532, 0.999\]

Text: "소득세 계산 방법..."

Metadata: {source: tax.docx}

------------------------------------------------------------------------

# 변경된 코드 --- LLM

Before:

``` python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
```

After:

``` python
from langchain_upstage import ChatUpstage

llm = ChatUpstage()
```

------------------------------------------------------------------------

# 내부 동작 차이 (LLM)

동작 동일:

Prompt

→ LLM

→ Answer

차이:

Language optimization

Upstage:

한국어 generation 성능 우수

------------------------------------------------------------------------

# Retrieval 동작은 동일

``` python
retrieved_docs = database.similarity_search(
    query,
    k=3
)
```

Pinecone에서 semantic search 수행

------------------------------------------------------------------------

# RetrievalQA Chain 변경 없음

``` python
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=database.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
```

------------------------------------------------------------------------

# 전체 아키텍처 (최종 구조)

User ↓ Upstage Embedding ↓ Pinecone Vector DB ↓ Retriever ↓ Prompt ↓
ChatUpstage ↓ Answer

------------------------------------------------------------------------

# Production에서 이 구조의 의미

이 구조는 실제 Production에서 가장 이상적인 구조 중 하나입니다.

이유:

Vector DB:

Pinecone

→ 확장성 → 안정성

Embedding:

Upstage

→ 한국어 retrieval 최적화

LLM:

Upstage Solar

→ 한국어 generation 최적화

------------------------------------------------------------------------

# Chroma vs Pinecone vs Upstage 조합 비교

개발 환경:

Chroma + OpenAI

운영 환경:

Pinecone + OpenAI

한국어 운영 환경:

Pinecone + Upstage (현재 구조)

------------------------------------------------------------------------

# 성능 영향

한국어 기준:

Retrieval Accuracy:

Upstage Embedding \> OpenAI Embedding

Generation Quality:

ChatUpstage \> ChatOpenAI

------------------------------------------------------------------------

# 결론

이번 실습의 핵심 변경:

Embedding:

OpenAI → Upstage

LLM:

ChatOpenAI → ChatUpstage

Vector DB:

Pinecone 유지

------------------------------------------------------------------------

이 구조는 한국어 Production RAG의 권장 구조입니다.
