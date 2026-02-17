# 실습 02 --- Retrieval Chain 변경 (RetrievalQA → create_retrieval_chain)

이 문서는 Base RAG 구조에서 **RetrievalQA에서 create_retrieval_chain으로
변경된 부분만 설명**합니다.

Base 구조 설명은 다음 문서를 기준으로 합니다:

00_base_rag_langchain_chroma_FULL.md

------------------------------------------------------------------------

# 변경 목적

기존 방식:

RetrievalQA

신규 방식:

create_retrieval_chain

이 변경은 LangChain 최신 구조(LCEL, Runnable 기반)에 맞춘 구조입니다.

RetrievalQA는 현재 deprecated 방향입니다.

------------------------------------------------------------------------

# 기존 구조 (RetrievalQA)

``` python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=database.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

ai_message = qa_chain.invoke({"query": query})
```

문제점:

-   내부 동작이 숨겨져 있음
-   커스터마이징 어려움
-   확장성 제한

------------------------------------------------------------------------

# 새로운 구조 (create_retrieval_chain)

구조를 2단계로 분리:

1.  combine_documents_chain
2.  retrieval_chain

------------------------------------------------------------------------

# 변경된 전체 코드

``` python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

combine_docs_chain = create_stuff_documents_chainAChain(
    llm,
    retrieval_qa_chat_prompt
)

retrieval_chain = create_retrieval_chain(
    retriever,
    combine_docs_chain
)

ai_message = retrieval_chain.invoke({"input": query})
```

------------------------------------------------------------------------

# 내부 동작 상세 설명

create_retrieval_chain 내부 동작:

Step 1:

User Query

↓

Retriever 실행

``` python
docs = retriever.invoke(query)
```

------------------------------------------------------------------------

Step 2:

Prompt 생성

``` python
prompt = retrieval_qa_chat_prompt.format(
    context=docs,
    input=query
)
```

------------------------------------------------------------------------

Step 3:

LLM 실행

``` python
response = llm.invoke(prompt)
```

------------------------------------------------------------------------

Step 4:

결과 반환

``` python
return response
```

------------------------------------------------------------------------

# create_stuff_documents_chain 역할

``` python
combine_docs_chain = create_stuff_documents_chain(
    llm,
    retrieval_qa_chat_prompt
)
```

역할:

retrieved docs + prompt + llm 연결

------------------------------------------------------------------------

# Retrieval Chain 전체 흐름

User Query ↓ Retriever ↓ Documents ↓ combine_docs_chain ↓ Prompt 생성 ↓
LLM ↓ Answer

------------------------------------------------------------------------

# 아키텍처 비교

기존:

User ↓ RetrievalQA ↓ Answer

내부 구조 숨김

------------------------------------------------------------------------

신규:

User ↓ Retriever ↓ combine_docs_chain ↓ LLM ↓ Answer

구조 명확

------------------------------------------------------------------------

# 왜 create_retrieval_chain이 중요한가

장점:

1.  내부 구조 명확

2.  커스터마이징 가능

예:

-   reranker 추가
-   hybrid search 추가
-   memory 추가

3.  production-ready 구조

------------------------------------------------------------------------

# 실무에서 create_retrieval_chain 사용 이유

실제 Production RAG는 다음 구조 사용:

Retriever

→ re-ranker

→ prompt builder

→ llm

→ output parser

create_retrieval_chain은 이를 구현 가능

------------------------------------------------------------------------

# prompt 변경

기존:

``` python
prompt = hub.pull("rlm/rag-prompt")
```

변경:

``` python
retrieval_qa_chat_prompt = hub.pull(
    "langchain-ai/retrieval-qa-chat"
)
```

차이:

retrieval-qa-chat:

chat model 최적화

------------------------------------------------------------------------

# retriever 생성 변경

``` python
retriever = database.as_retriever(
    search_kwargs={"k": 1}
)
```

의미:

top 1 document만 retrieval

------------------------------------------------------------------------

# 결론

변경된 핵심:

RetrievalQA

→

create_retrieval_chain

------------------------------------------------------------------------

이 변경은 LangChain 최신 표준 구조입니다.

이후 모든 RAG 실습은 create_retrieval_chain 기반으로 진행됩니다.
