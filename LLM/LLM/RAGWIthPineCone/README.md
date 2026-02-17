# 실습 03 --- Vector DB 변경 (Chroma ➜ Pinecone)

이 문서는 **Base RAG(Chroma 기반)** 구조를 기준으로, **Vector Database만
Pinecone으로 변경**한 부분을 "변경점 중심"으로 정리합니다.\
(Loader/Splitter/Embedding/Retrieval/Prompt/LLM/Chain의 기본 개념은
`00_base_rag_langchain_chroma_FULL.md`를 기준으로 합니다.)

------------------------------------------------------------------------

## 변경 요약 (한눈에 보기)

  -----------------------------------------------------------------------
  구분                    Before (로컬/개발)      After (클라우드/운영)
  ----------------------- ----------------------- -----------------------
  Vector DB               Chroma (로컬 디렉토리   Pinecone (관리형 Vector
                          persist)                DB)

  저장 방식               로컬 파일 기반          원격 인덱스 기반

  확장성                  단일 머신/개발에 적합   멀티 인스턴스/운영에
                                                  적합

  주요 장점               설치/실행 간단, 빠른    운영 안정성, 확장성, 팀
                          실험                    공유

  주요 주의점             로컬 디스크 의존        인덱스/차원/키 관리
                                                  필요(비용 포함)
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 1) 변경 목적

Chroma는 로컬 개발/PoC에 매우 편하지만, 운영 환경에서 다음 한계가
생깁니다.

-   서버가 여러 대(멀티 인스턴스)면 **Vector DB를 공유**하기 어렵다
-   컨테이너/서버 재배포 시 **로컬 persist 관리가 복잡**
-   운영 수준의 **SLA/모니터링/확장성**이 부족할 수 있다

Pinecone으로 바꾸면: - Vector DB가 **클라우드 서비스로 분리**되어 여러
애플리케이션 서버가 동일 인덱스를 사용 - 운영/확장/가용성 측면에서
안정적

------------------------------------------------------------------------

# 2) 핵심 변경 포인트

Base(Chroma)에서 바뀐 건 딱 2가지입니다.

1)  **Vector DB 생성/로드 코드**
2)  **환경변수(PINECONE_API_KEY) 및 index_name 관리**

나머지(문서 로드/스플릿/임베딩/프롬프트/LLM/체인)는 동일합니다.

------------------------------------------------------------------------

# 3) Pinecone 세팅 체크리스트 (실무 필수)

## 3.1 환경변수

`.env` 또는 시스템 환경변수로 다음이 있어야 합니다.

``` env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
```

> 강의 코드에서는 `load_dotenv()` 호출 후
> `os.environ.get("PINECONE_API_KEY")`로 가져옵니다.

## 3.2 인덱스(index) 개념

Pinecone에서 Vector는 **Index**에 저장됩니다.

-   `index_name = "tax-index"` 같은 이름으로 인덱스를 지정
-   같은 인덱스는 여러 서버/사용자가 공유 가능
-   인덱스는 **dimension(벡터 차원)**, metric(유사도 지표) 등이 맞아야
    합니다.

⚠️ 중요한 실무 포인트\
Embedding 모델이 바뀌면 벡터 차원이 바뀔 수 있습니다.\
→ Pinecone 인덱스 dimension이 embedding output dimension과 **일치**해야
합니다.

(OpenAI `text-embedding-3-large`의 차원은 Pinecone 인덱스와 일치해야
합니다. 차원 값은 모델/버전에 따라 달라질 수 있으니 "내가 쓰는
embedding의 실제 dimension"을 확인한 뒤 인덱스를 맞추는 습관이
필요합니다.)

------------------------------------------------------------------------

# 4) 변경된 코드 (Vector DB 파트만)

## 4.1 패키지 변경

Chroma 실습 대비, Pinecone 관련 패키지가 추가됩니다.

``` python
%pip install langchain langchain-core langchain-community langchain-text-splitters langchain-openai langchain-pinecone docx2txt
```

핵심 추가: - `langchain-pinecone` - `pinecone` (내부에서 사용)

------------------------------------------------------------------------

## 4.2 문서 로드/스플릿/임베딩 (동일)

> 이 파트는 Base와 동일하므로 "변경 없음"으로 취급합니다.

``` python
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)

loader = Docx2txtLoader('./tax.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
```

------------------------------------------------------------------------

## 4.3 Chroma → Pinecone로 변경된 핵심 코드

### Before (Chroma)

``` python
from langchain_chroma import Chroma

database = Chroma.from_documents(
    documents=document_list,
    embedding=embedding,
    collection_name='chroma-tax',
    persist_directory="./chroma"
)
```

### After (Pinecone)

``` python
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

index_name = 'tax-index'
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

database = PineconeVectorStore.from_documents(
    document_list,
    embedding,
    index_name=index_name
)
```

### 이 코드가 하는 일(실제 동작)

-   `from_documents(...)`는 내부적으로:
    1)  각 문서 chunk를 embedding으로 벡터화
    2)  (text + metadata + vector)를 Pinecone index에 upsert(저장)
-   결과적으로 `database`는 "Pinecone 인덱스를 바라보는 LangChain
    VectorStore 래퍼"가 됩니다.

⚠️ 실무 주의\
Pinecone 인덱스가 자동 생성되는지/미리 만들어야 하는지는 설정/버전/계정
구성에 따라 달라질 수 있습니다.\
안전한 운영 관점에서는: - **인덱스를 미리 생성**하고 (dimension/metric
명시) - 애플리케이션에서는 **이미 존재하는 인덱스에 upsert**하는 패턴을
권장합니다.

------------------------------------------------------------------------

# 5) Retriever 생성 변경점

Vector DB가 Pinecone으로 바뀌었지만, Retriever 인터페이스는 동일합니다.

``` python
retriever = database.as_retriever(search_kwargs={'k': 4})
retriever.invoke(query)
```

-   `k`: top-k 문서 몇 개를 가져올지
-   `invoke(query)`: 실제로 어떤 문서가 검색되는지 확인 (디버깅용으로
    매우 중요)

------------------------------------------------------------------------

# 6) Prompt / LLM / Chain (변경 없음)

``` python
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o')

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

ai_message = qa_chain.invoke({"query": query})
ai_message
```

> 이 실습은 "Vector DB만 교체"가 목적이라 RetrievalQA를 그대로
> 사용했지만,\
> 다음 실습들에서는 `create_retrieval_chain` 기반으로 통일하는 방향이 더
> 최신 표준입니다.

------------------------------------------------------------------------

# 7) 실무 관점: Chroma vs Pinecone 선택 기준

## Chroma를 추천하는 경우

-   로컬 개발 / 빠른 PoC
-   인덱스 비용 없이 실험
-   팀 공유/운영이 아직 아닌 단계

## Pinecone을 추천하는 경우

-   운영(Production) 환경
-   여러 서버/여러 서비스가 동일 인덱스를 공유해야 함
-   인덱스 가용성/확장성/운영 편의가 중요
-   데이터가 커지고 검색 트래픽이 증가

------------------------------------------------------------------------

# 8) 운영에서 반드시 고려할 것 (체크리스트)

-   [ ] `PINECONE_API_KEY` 보안 관리(Secret Manager/Key Vault 등)
-   [ ] 인덱스 dimension이 embedding output과 일치하는지
-   [ ] 인덱스 비용(저장량 + 검색 트래픽)
-   [ ] 문서 업데이트 전략(재업서트, 버전 관리, 삭제 전략)
-   [ ] metadata 설계(문서 출처/버전/권한 등)
-   [ ] 멀티테넌시/권한(사내 문서 권한 모델과 연결)

------------------------------------------------------------------------

# 결론

이번 실습에서 바뀐 건 "Vector DB 구현체"뿐입니다.

-   Chroma(로컬) → Pinecone(클라우드)
-   나머지 RAG 흐름(Loader/Splitter/Embedding/Retriever/Prompt/LLM)은
    동일
-   운영으로 갈수록 Pinecone 같은 관리형 Vector DB가 현실적인 선택이
    됩니다.
