# 실습 05 --- Retrieval 효율 개선 (Keyword Dictionary + LCEL Chain)

이 문서는 기존 Pinecone 기반 RAG 구조에서 **Retrieval 정확도를 개선하기
위해 Keyword Dictionary Chain을 추가한 부분만 설명**합니다.

기준 문서:

-   Base RAG 구조: 00_base_rag_langchain_chroma_FULL.md
-   Pinecone Vector DB: 03_vector_db_chroma_to_pinecone.md
-   Pinecone + Upstage 구조: 04_vector_db_pinecone_upstage.md

------------------------------------------------------------------------

# 변경 목적

Semantic Search는 의미 기반 검색이지만, domain-specific 용어 차이로 인해
retrieval 실패가 발생할 수 있습니다.

예:

Knowledge Base: "거주자의 종합소득세"

User Question: "사람의 종합소득세"

Semantic mismatch 발생 가능

------------------------------------------------------------------------

# 해결 방법 --- Keyword Dictionary Chain

Keyword 사전을 활용하여 User Query를 Domain Query로 변환합니다.

예:

사전:

사람 → 거주자

------------------------------------------------------------------------

# 변경된 코드

## Dictionary Chain 생성

``` python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = [
    "사람을 나타내는 표현 -> 거주자"
]

prompt = ChatPromptTemplate.from_template(
"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
변경이 필요 없다면 질문을 그대로 리턴해주세요.

사전: {dictionary}

질문: {question}
"""
)

dictionary_chain = prompt | llm | StrOutputParser()
```

------------------------------------------------------------------------

# Chain 연결 (LCEL)

``` python
tax_chain = {"query": dictionary_chain} | qa_chain
```

Execution Flow:

User Question\
→ dictionary_chain\
→ Modified Question\
→ qa_chain\
→ Answer

------------------------------------------------------------------------

# 실행

``` python
ai_response = tax_chain.invoke({
    "question": query
})
```

------------------------------------------------------------------------

# 내부 동작

Step 1 --- User Question 입력

Step 2 --- dictionary_chain 실행

Step 3 --- Domain-specific Query 생성

Step 4 --- Retriever 실행

Step 5 --- Pinecone Vector Search

Step 6 --- LLM Answer 생성

------------------------------------------------------------------------

# LCEL (LangChain Expression Language)

Chain 연결 방식:

``` python
chainA | chainB | chainC
```

Data flow 자동 연결

------------------------------------------------------------------------

# 왜 중요한가 (Production 관점)

Enterprise Knowledge Base는 다음 문제가 존재:

동의어 문제:

고객 vs 회원 vs 사용자

Dictionary Chain으로 해결 가능

------------------------------------------------------------------------

# 성능 개선 효과

Retrieval Recall 증가

Retrieval Precision 증가

Semantic mismatch 해결

------------------------------------------------------------------------

# 최종 아키텍처

User Question ↓ Dictionary Chain ↓ Retriever ↓ Pinecone ↓ LLM ↓ Answer

------------------------------------------------------------------------

# 결론

추가된 구성요소:

Dictionary Chain

역할:

User Query → Domain Query 변환

------------------------------------------------------------------------

이 구조는 Production RAG에서 Retrieval 정확도 향상을 위한 핵심
기술입니다.
