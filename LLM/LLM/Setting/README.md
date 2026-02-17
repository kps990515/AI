# LangChain LLM 환경 설정 및 Provider 검증 가이드

## 실습 주제

이 문서는 LangChain 기반 LLM Application 개발을 시작하기 위한 첫 단계로,
다음 3가지 Provider의 정상 동작을 검증하는 것이 목적입니다.

-   ChatOpenAI (OpenAI API 기반)
-   ChatUpstage (Upstage API 기반)
-   ChatOllama (로컬 LLM 기반)

이 단계는 이후 RAG(Retrieval Augmented Generation)를 구현하기 위한 필수
사전 단계입니다.

------------------------------------------------------------------------

## 왜 이 단계가 중요한가

RAG 시스템은 다음 구조로 동작합니다:

User Question ↓ Retriever ↓ Vector Store ↓ Relevant Documents ↓ Prompt
Template ↓ LLM ↓ Final Answer

이 중에서 LLM이 정상 동작하지 않으면 전체 RAG 시스템이 작동하지
않습니다.

따라서 먼저 LLM 호출 환경을 안정적으로 구축해야 합니다.

------------------------------------------------------------------------

## 가상환경 설정

### 폴더 생성

``` bash
mkdir llm-application
cd llm-application
```

### pyenv virtualenv 생성

``` bash
pyenv virtualenv 3.10 llm-application
pyenv local llm-application
# 아마 안될거라서 claude에 해달라고 하기
# bash에서 해당폴더 접근시 가상환경 자동실행까지 부탁하기
```

자동 활성화를 위해 `.python-version` 파일이 생성됩니다.

Windows에서는 pyenv 대신 venv 사용을 권장합니다.

------------------------------------------------------------------------

## 환경변수 설정

`.env` 파일 생성:

``` env
OPENAI_API_KEY=your_key
UPSTAGE_API_KEY=your_key
```

Python 코드:

``` python
from dotenv import load_dotenv
load_dotenv()
```

------------------------------------------------------------------------

## 패키지 설치

``` python
%pip install python-dotenv langchain-openai langchain-upstage langchain-community
```

------------------------------------------------------------------------

## OpenAI 검증 코드

``` python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4o-mini")
ai_message = llm.invoke("인프런에 어떤 강의가 있나요?")

print(ai_message.content)
```

------------------------------------------------------------------------

## Upstage 검증 코드

``` python
from langchain_upstage import ChatUpstage

llm = ChatUpstage()
ai_message = llm.invoke("인프런에 어떤 강의가 있나요?")

print(ai_message.content)
```

------------------------------------------------------------------------

## Ollama 검증 코드

모델 다운로드:

``` bash
ollama pull gemma2
```

Python 코드:

``` python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="gemma2")
ai_message = llm.invoke("인프런에 어떤 강의가 있나요?")

print(ai_message.content)
```

------------------------------------------------------------------------

## Provider 비교

  Provider   API Key 필요   네트워크 필요   로컬 실행
  ---------- -------------- --------------- -----------
  OpenAI     Yes            Yes             No
  Upstage    Yes            Yes             No
  Ollama     No             No              Yes

------------------------------------------------------------------------

## 결론

이 단계가 완료되면:

-   LangChain 환경 구축 완료
-   LLM 호출 검증 완료
-   RAG 실습 준비 완료

다음 단계에서는 Vector Store, Embedding, Retriever를 구성하여 완전한 RAG
시스템을 구현합니다.
