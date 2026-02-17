# LangChain 핵심 요약 (실무용 빠른 복습 가이드)

이 문서는 LangChain 1\~5강의 핵심 개념을 실무 기준으로 빠르게 복습할 수
있도록 요약한 문서입니다.

목표:

-   LangChain 핵심 구조 이해
-   LLM 호출 흐름 이해
-   Prompt → LLM → Parser → Chain 전체 흐름 이해
-   실무에서 사용하는 LCEL 파이프라인 이해

------------------------------------------------------------------------

# 1. LangChain 전체 구조 한눈에 보기

LangChain의 기본 파이프라인:

    Input
     ↓
    PromptTemplate
     ↓
    LLM
     ↓
    OutputParser
     ↓
    Final Output

확장 구조:

    Input
     ↓
    Chain
     ↓
    LLM
     ↓
    Structured Output
     ↓
    Application Logic

------------------------------------------------------------------------

# 2. 핵심 구성요소 요약

## 2.1 LLM

역할:

자연어 입력 → 자연어 출력

예:

``` python
llm = ChatOllama(model="llama3.2:1b")
response = llm.invoke("What is the capital of France?")
```

출력:

    AIMessage

실제 답변:

    response.content

------------------------------------------------------------------------

## 2.2 PromptTemplate

역할:

프롬프트 템플릿 생성

예:

``` python
prompt = PromptTemplate(
    template="Capital of {country}?",
    input_variables=["country"]
)
```

실행:

``` python
prompt.invoke({"country": "France"})
```

출력:

    "Capital of France?"

------------------------------------------------------------------------

## 2.3 ChatPromptTemplate

역할:

SystemMessage / HumanMessage 구조 생성

예:

``` python
ChatPromptTemplate.from_messages([
    ("system", "You are helpful"),
    ("human", "{question}")
])
```

------------------------------------------------------------------------

## 2.4 Message 종류

  Type            역할
  --------------- ----------------
  SystemMessage   모델 행동 정의
  HumanMessage    사용자 입력
  AIMessage       모델 응답

------------------------------------------------------------------------

## 2.5 OutputParser

역할:

LLM 출력 변환

종류:

### StrOutputParser

``` python
parser = StrOutputParser()
```

출력:

    str

------------------------------------------------------------------------

### Structured Output (Pydantic)

``` python
class Country(BaseModel):
    capital: str
```

``` python
structured_llm = llm.with_structured_output(Country)
```

출력:

    Country object

------------------------------------------------------------------------

## 2.6 Runnable

Runnable은 invoke 가능한 객체

예:

    PromptTemplate
    LLM
    OutputParser
    Chain

------------------------------------------------------------------------

# 3. LCEL (LangChain Expression Language)

핵심 문법:

    |

의미:

출력 → 다음 입력

예:

``` python
chain = prompt | llm | parser
```

실행:

``` python
chain.invoke({"country": "France"})
```

------------------------------------------------------------------------

# 4. Chain 연결

예:

    country → food → recipe

코드:

``` python
final_chain = {"food": food_chain} | recipe_chain
```

흐름:

    Input
     ↓
    food_chain
     ↓
    recipe_chain
     ↓
    Output

------------------------------------------------------------------------

# 5. RunnablePassthrough

역할:

입력 그대로 전달

예:

``` python
RunnablePassthrough()
```

------------------------------------------------------------------------

# 6. Temperature

역할:

출력 랜덤성 제어

  Temperature   특징
  ------------- ---------------
  0             deterministic
  0.7           일반
  1             creative

실무 권장:

    0~0.2

------------------------------------------------------------------------

# 7. 실무에서 사용하는 기본 구조

가장 기본:

``` python
chain = prompt | llm | parser
chain.invoke(input)
```

고급:

``` python
final_chain = {"x": chain1} | chain2
```

------------------------------------------------------------------------

# 8. 전체 실행 흐름 다이어그램

    User Input
     ↓
    PromptTemplate
     ↓
    LLM
     ↓
    OutputParser
     ↓
    Chain
     ↓
    Final Output

------------------------------------------------------------------------

# 9. 실무 기준 핵심 포인트

반드시 기억할 것:

1.  PromptTemplate → 입력 생성

2.  LLM → 답변 생성

3.  OutputParser → 출력 정리

4.  LCEL → 체인 연결

5.  invoke() → 실행

------------------------------------------------------------------------

# 10. 실무에서 가장 많이 사용하는 패턴

패턴 1

``` python
prompt | llm | parser
```

패턴 2

``` python
{"key": chain1} | chain2
```

패턴 3

``` python
llm.with_structured_output(Model)
```

------------------------------------------------------------------------

# 11. 최종 핵심 요약

LangChain은 다음을 쉽게 만든다:

-   LLM 호출
-   Prompt 관리
-   출력 파싱
-   Chain 연결
-   파이프라인 구성

핵심 실행:

    chain.invoke(input)

이 한 줄로 전체 파이프라인 실행 가능

------------------------------------------------------------------------

# 12. 실무에서의 실제 구조

    User Input
     ↓
    PromptTemplate
     ↓
    LLM
     ↓
    Parser
     ↓
    Business Logic
     ↓
    API Response

LangChain은 LLM 기반 백엔드 파이프라인을 구축하기 위한 프레임워크이다.
