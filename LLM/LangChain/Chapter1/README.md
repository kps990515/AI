# 실습 주제

-   **한 줄 요약:** LangChain의 ChatModel(LLM) 래퍼를 사용해
    로컬(Ollama) 및 클라우드(OpenAI / Azure OpenAI) 모델을 동일한
    인터페이스(invoke)로 호출해 답변을 생성한다.
-   **해결하려는 문제:**
    -   LLM 제공자가 달라져도 호출 방식이 제각각이면 코드 유지보수가
        어려워진다.\
    -   LangChain은 각 LLM 벤더(OpenAI, Azure, Ollama 등)의 차이를
        추상화하여 동일한 방식으로 호출할 수 있도록 한다.\
    -   또한 .env 환경 변수를 사용해 API Key 같은 민감한 정보를 안전하게
        관리할 수 있다.

------------------------------------------------------------------------

# 전체 구조 설명

## 1) 코드의 전체 흐름

### Step 1. 라이브러리 설치

필요한 LangChain 패키지와 dotenv를 설치한다.

``` python
%pip install -q langchain-ollama langchain-openai langchain-anthropic python-dotenv
```

역할:

-   langchain-ollama: Ollama 로컬 모델 사용
-   langchain-openai: OpenAI 및 Azure OpenAI 사용
-   langchain-anthropic: Anthropic Claude 모델 사용 가능
-   python-dotenv: 환경 변수 관리

------------------------------------------------------------------------

### Step 2. 환경 변수 로딩

``` python
from dotenv import load_dotenv
load_dotenv()
```

역할:

-   .env 파일의 내용을 시스템 환경 변수로 로드
-   API 키를 코드에 직접 작성하지 않아도 되도록 함

예시 .env 파일:

    OPENAI_API_KEY=your_key_here
    AZURE_OPENAI_API_KEY=your_key_here
    AZURE_OPENAI_ENDPOINT=https://xxxx.openai.azure.com/

------------------------------------------------------------------------

### Step 3. Ollama 로컬 LLM 사용

``` python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:1b")
response = llm.invoke("What is the capital of France?")
```

역할:

-   로컬에 설치된 llama 모델을 사용
-   invoke()는 LangChain 표준 실행 함수

결과 타입:

    AIMessage

실제 답변:

    response.content

------------------------------------------------------------------------

### Step 4. OpenAI GPT 사용

``` python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("What is the capital of France?")
```

필요 조건:

-   OPENAI_API_KEY 환경 변수 필요

역할:

-   OpenAI 서버에 요청
-   GPT 모델이 답변 생성

------------------------------------------------------------------------

### Step 5. Azure OpenAI 사용

``` python
from langchain_openai import ChatAzureOpenAI

llm = ChatAzureOpenAI(model="gpt-4o-mini")
response = llm.invoke("What is the capital of France?")
```

필요 조건:

-   AZURE_OPENAI_API_KEY
-   AZURE_OPENAI_ENDPOINT
-   AZURE_OPENAI_API_VERSION

역할:

-   Azure OpenAI 서비스 호출

------------------------------------------------------------------------

# LangChain 주요 개념 설명

## LLM (Large Language Model)

역할:

-   자연어를 입력받아 자연어를 생성

LangChain에서의 구현 클래스:

-   ChatOllama
-   ChatOpenAI
-   ChatAzureOpenAI

공통 사용 방법:

``` python
llm.invoke("질문")
```

# 핵심 요약

LangChain의 ChatModel을 사용하면

-   로컬 모델
-   OpenAI
-   Azure OpenAI

를 동일한 방식으로 사용할 수 있다.

핵심 실행 함수:

    llm.invoke()

핵심 장점:

-   벤더 변경 쉬움
-   코드 일관성
-   확장성 높음
