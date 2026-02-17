# 실습 주제

-   **한 줄 요약:** LangChain의 PromptTemplate 및 ChatPromptTemplate을
    사용하여 LLM에 전달할 프롬프트를 구조화하고 변수 기반으로 동적으로
    생성하는 방법을 학습한다.
-   **해결하려는 문제:**
    -   프롬프트를 코드에 하드코딩하면 재사용이 어렵고 유지보수가
        힘들다.
    -   LangChain의 PromptTemplate을 사용하면 변수 기반으로 프롬프트를
        재사용할 수 있고, ChatPromptTemplate을 통해 시스템/사용자/AI
        메시지 기반의 구조화된 프롬프트를 쉽게 생성할 수 있다.

------------------------------------------------------------------------

# 전체 구조 설명

이 강의에서는 LangChain의 핵심 구성 요소 중 하나인 PromptTemplate과
ChatPromptTemplate을 사용하여 LLM에 전달할 프롬프트를 구조화하는 방법을
학습합니다.

전체 흐름:

1.  LLM(ChatOllama) 객체 생성
2.  PromptTemplate을 사용한 문자열 기반 프롬프트 생성
3.  메시지 기반 프롬프트 생성(SystemMessage, HumanMessage, AIMessage)
4.  ChatPromptTemplate을 사용한 채팅형 프롬프트 템플릿 생성
5.  생성된 프롬프트를 LLM.invoke()로 전달하여 응답 생성

------------------------------------------------------------------------

# LangChain 주요 개념 설명

## LLM (Large Language Model)

역할:

-   프롬프트를 입력받아 답변 생성

사용 방법:

``` python
from langchain_ollama import ChatOllama

# Ollama를 이용한 로컬 LLM 설정
llm = ChatOllama(model="llama3.2:1b")
```

입력:

-   string
-   PromptValue
-   message list

출력:

    AIMessage

------------------------------------------------------------------------

## PromptTemplate

역할:

-   변수 기반 문자열 프롬프트 생성

기본 구조:

``` python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 변수가 포함된 템플릿 정의
prompt_template = PromptTemplate(
    template="What is the capital of {country}? Return the name of the city only",
    input_variables=["country"],
)

# 아래처럼 해도 위와 같은 결과를 얻을 수 있습니다.
prompt_template.from_template("What is the capital of {country}? Return the name of the city only")
```

변수 주입:

``` python
prompt = prompt_template.invoke({
    "country": "France"
})
```

결과:

    "What is the capital of France?"

장점:

-   재사용 가능
-   동적 프롬프트 생성 가능
-   유지보수 용이

------------------------------------------------------------------------

## ChatPromptTemplate

역할:

-   채팅 기반 프롬프트 생성
-   System, Human, AI message 형태 지정가능

구조:

``` python
from langchain_core.prompts import ChatPromptTemplate

# 채팅 형식의 프롬프트 템플릿 생성
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant!"), #시스템 프롬프트
    ("human", "What is the capital of {country}?"), #사용자 질문
])


# 템플릿에 변수 값 주입
chat_prompt = chat_prompt_template.invoke({"country": "France"})

print(chat_prompt)
```

변수 주입:

``` python
chat_prompt = chat_prompt_template.invoke({
    "country": "France"
})
```

결과출력:

    # llm.invoke(chat_prompt)

------------------------------------------------------------------------

## Runnable

PromptTemplate과 ChatPromptTemplate은 Runnable 객체입니다.

따라서 다음과 같이 실행 가능합니다:

``` python
prompt_template.invoke()
chat_prompt_template.invoke()
```

------------------------------------------------------------------------

# 코드 흐름 상세 분석

## Step 1. LLM 생성

``` python
llm = ChatOllama(model="llama3.2:1b")
```

역할:

-   Ollama 로컬 LLM 객체 생성

------------------------------------------------------------------------

## Step 2. PromptTemplate 생성

``` python
prompt_template = PromptTemplate(
    template="What is the capital of {country}? Return the name of the city only",
    input_variables=["country"],
)
```

역할:

-   country 변수를 받는 템플릿 생성

------------------------------------------------------------------------

## Step 3. 변수 주입

``` python
prompt = prompt_template.invoke({
    "country": "France"
})
```

결과:

    "What is the capital of France?"

------------------------------------------------------------------------

## Step 4. LLM 호출

``` python
llm.invoke(prompt)
```

결과:

    AIMessage(content="The capital of France is Paris.")

------------------------------------------------------------------------

## Step 5. 메시지 기반 프롬프트

``` python
message_list = [
    SystemMessage(...),
    HumanMessage(...),
]
```

LLM 호출:

``` python
llm.invoke(message_list)
```

------------------------------------------------------------------------

## Step 6. ChatPromptTemplate 사용

``` python
chat_prompt_template = ChatPromptTemplate.from_messages(...)
chat_prompt = chat_prompt_template.invoke({
    "country": "France"
})
```

LLM 호출:

``` python
llm.invoke(chat_prompt)
```

------------------------------------------------------------------------

# 핵심 요약

PromptTemplate:

-   문자열 기반 템플릿

ChatPromptTemplate:

-   메시지 기반 템플릿

핵심 실행:

    prompt_template.invoke()
    chat_prompt_template.invoke()
    llm.invoke()

장점:

-   재사용 가능
-   유지보수 용이
-   동적 프롬프트 생성
