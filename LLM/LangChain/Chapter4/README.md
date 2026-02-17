# 실습 주제

-   **한 줄 요약:** LCEL(LangChain Expression Language)의 파이프
    연산자(`|`)와 `RunnablePassthrough`를 사용해 **Prompt → LLM →
    Parser**를 연결하고, 여러 단계를 조합한 **복합 체인**을 만든다.
-   **해결하려는 문제:**
    -   프롬프트 생성 → LLM 호출 → 결과 파싱 같은 단계를 "수동으로"
        이어붙이면 코드가 길어지고 재사용이 어렵다.
    -   LCEL은 각 컴포넌트를 **Runnable**로 보고, `|`로 조립해 **데이터
        흐름이 명확한 파이프라인(체인)** 을 만든다.
    -   `RunnablePassthrough` / dict 조합을 이용하면 입력을
        분기·병합하여 "다단계 추론" 같은 복잡한 흐름도 깔끔하게 구성할
        수 있다.

------------------------------------------------------------------------

# 전체 구조 설명

이 노트북은 3강에서 했던 "PromptTemplate + LLM + OutputParser"를 **LCEL
문법**으로 연결하는 것이 핵심입니다.

크게 두 가지를 합니다.

1)  **간단 체인**: `prompt_template | llm | output_parser`
2)  **복합 체인(다단계 추론)**:
    -   먼저 정보로부터 "국가(country)"를 추론하고
    -   그 국가를 이용해 "수도(capital)"를 다시 묻는 체인으로 연결

전체 흐름:

1.  `ChatOllama`로 LLM 초기화
2.  `PromptTemplate` + `StrOutputParser` 준비
3.  LCEL 파이프(`|`)로 **capital_chain** 구성 후 실행
4.  국가 추측용 `country_prompt` + 체인(**country_chain**) 구성
5.  `RunnablePassthrough` + dict 파이프를 이용해 **final_chain** 구성
6.  최종 입력(information, continent) → 국가 추론 → 수도 반환

------------------------------------------------------------------------

# LangChain 주요 개념 설명 (이 실습 기준)

## LLM

-   **역할:** 프롬프트를 입력받아 답변 생성
-   **이 실습에서 사용:**

``` python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2:1b")
```

-   기본 출력은 `AIMessage`

------------------------------------------------------------------------

## PromptTemplate

-   **역할:** 입력 dict를 받아 템플릿을 채워 프롬프트(또는
    PromptValue)를 만든다.
-   **예:**

``` python
prompt_template = PromptTemplate(
    template="What is the capital of {country}? Return the name of the city only",
    input_variables=["country"],
)
```

-   LCEL에서 `prompt_template | llm`처럼 바로 연결되는 이유:
    -   `PromptTemplate`도 Runnable처럼 동작하며 `.invoke()`가 가능하기
        때문

------------------------------------------------------------------------

## OutputParser

-   **역할:** LLM 출력(`AIMessage`)을 우리가 원하는 타입으로 변환
-   **이 실습에서 사용:**

``` python
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
```

-   `StrOutputParser`는 `AIMessage.content`를 꺼내 `str`로 만든다.

------------------------------------------------------------------------

## Runnable (핵심)

-   **역할:** `.invoke()`로 실행 가능한 단위
-   LCEL은 "모든 컴포넌트를 Runnable로 보고 조립"한다.
-   이 실습에서 Runnable인 것들:
    -   `prompt_template`
    -   `llm`
    -   `output_parser`
    -   `RunnablePassthrough()`
    -   `capital_chain`, `country_chain`, `final_chain` (조립 결과물도
        Runnable)

즉, 체인도 결국 Runnable이기 때문에:

``` python
capital_chain.invoke(...)
final_chain.invoke(...)
```

가 된다.

------------------------------------------------------------------------

## Chain

-   LCEL에서 "체인"은 대체로 Runnable 파이프라인 자체를 의미한다.
-   `a | b | c`는 내부적으로 "a의 출력이 b의 입력으로, b의 출력이 c의
    입력으로" 흐르는 **RunnableSequence** 같은 개념으로 볼 수 있다.

------------------------------------------------------------------------

# 코드 흐름 상세 분석

## Step 1. LLM 초기화

``` python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2:1b")
```

-   이후 모든 체인은 이 `llm`을 공유한다.

------------------------------------------------------------------------

## Step 2. Prompt + LLM + Parser를 수동으로 호출(복습)

(2강/3강 방식)

``` python
prompt = prompt_template.invoke({"country": "France"})
ai_message = llm.invoke(prompt)
answer = output_parser.invoke(ai_message)
```

문제점: - 단계가 많아질수록 반복 코드가 늘어난다. - 각 단계 연결이
명시적으로 "파이프" 형태로 보이지 않는다.

이 문제를 LCEL이 해결한다.

------------------------------------------------------------------------

## Step 3. LCEL로 간단 체인 만들기

### 3-1) 파이프 연결

``` python
capital_chain = prompt_template | llm | output_parser
```

이 한 줄은 아래를 의미한다:

1.  입력 dict를 `prompt_template`에 전달 → 프롬프트 생성
2.  그 프롬프트를 `llm`에 전달 → `AIMessage` 생성
3.  그 `AIMessage`를 `output_parser`에 전달 → `str` 생성

### 3-2) 체인 실행

``` python
capital_chain.invoke({"country": "France"})
# 'Paris.'
```

여기서 중요한 점: - 이제 `invoke`의 입력은 **prompt_template의 입력**과
동일한 dict 형태 - 결과는 **output_parser의 출력**과 동일한 타입(str)

즉, 파이프라인의 "입출력 타입"은 맨 앞/맨 뒤에 의해 결정된다.

------------------------------------------------------------------------

## Step 4. 복잡한 체인(다단계 추론) 구성

여기서는 "추론을 두 번" 한다.

1)  정보 + 대륙 → 국가 추론 (country_chain)
2)  국가 → 수도 추론 (capital_chain)

------------------------------------------------------------------------

### 4-1) 국가 추론용 프롬프트 및 체인

``` python
country_prompt = PromptTemplate(
    template="""Guess the name of the country in the {continent} based on the following information:
    {information}
    Return the name of the country only
    """,
    input_variables=["information", "continent"],
)

country_chain = country_prompt | llm | output_parser
```

입력:

``` python
{"information": "...", "continent": "Europe"}
```

출력:

    "France" (같은 국가명 문자열)

즉, `country_chain`의 결과는 "country 문자열"이다.

------------------------------------------------------------------------

### 4-2) RunnablePassthrough로 입력을 보존하면서 dict 구성

``` python
from langchain_core.runnables import RunnablePassthrough
```

`RunnablePassthrough()`는 "입력을 그대로 출력하는 Runnable"이다.

즉,

-   입력이 `"hello"`면 출력도 `"hello"`
-   입력이 `{"a": 1}`면 출력도 `{"a": 1}`

이걸 왜 쓰나?

-   LCEL에서 dict 형태 파이프를 만들 때,
-   특정 키에 "현재 입력을 그대로" 넣어야 하는 경우가 많다.

------------------------------------------------------------------------

### 4-3) 최종 체인(final_chain) 구성: 입력 분기 → country 생성 → capital 계산

``` python
final_chain = {
  "information": RunnablePassthrough(),
  "continent": RunnablePassthrough()
} | {
  "country": country_chain
} | capital_chain
```

이 한 줄을 단계별로 풀어쓰면 다음과 같다.

#### 단계 A) 입력을 원하는 키로 분해/보존

입력:

``` python
{"information": "...", "continent": "Europe"}
```

`{"information": RunnablePassthrough(), "continent": RunnablePassthrough()}`

-   `"information"` 키에는 입력 전체를 passthrough한 결과가 들어가는데,
    실제로는 "입력 dict에서 information/continent를 그대로 다음으로
    전달"하는 형태로 사용된다.
-   (실무에서는 종종 `itemgetter("information")` 같은 키 추출과 함께
    사용한다.)

이 실습에서는 "입력 dict를 그대로 유지한 채 다음 단계로 전달"하는 목적에
가깝다.

#### 단계 B) country_chain으로 국가 추론해서 {"country": ...} 만들기

`| {"country": country_chain}`

-   country_chain은 입력(dict)을 받아 "국가 문자열"을 출력한다.
-   그래서 결과는:

``` python
{"country": "France"}
```

#### 단계 C) capital_chain에 {"country": ...} 입력해서 수도 얻기

마지막 `| capital_chain`

-   capital_chain은 `{country: ...}`를 받아 수도 문자열을 반환한다.
-   최종 결과는 `"Paris."` 또는 `"Madrid"` 같은 문자열

------------------------------------------------------------------------

### 4-4) 최종 체인 실행

``` python
final_chain.invoke({
  "information": "This country is very famous for its wine in Europe",
  "continent": "Europe"
})
# 'Madrid'
```

이 결과는 모델이 "wine in Europe"를 보고 스페인(Spain)을 떠올렸기
때문에, capital_chain이 Madrid를 반환한 것으로 해석할 수 있다.

> 실무 포인트: 이런 "추론 체인"은 모델의 추론에 따라 결과가 달라질 수
> 있으며, 신뢰해야 하는 사실 데이터라면 Retriever/RAG 또는 외부 데이터
> 소스가 필요하다.

------------------------------------------------------------------------

# LCEL에서 dict 파이프가 의미하는 것 (중요)

LCEL에서 dict를 쓰면 "여러 Runnable의 결과를 모아서 dict로 만든다"는
의미로 이해하면 된다.

예:

``` python
{"a": runnable1, "b": runnable2}
```

-   입력을 runnable1과 runnable2에 각각 전달해
-   결과를 {"a": result1, "b": result2}로 합친다.

이게 병렬/분기 구조의 기본이다.

------------------------------------------------------------------------

# 실행 흐름 다이어그램 (텍스트 기반)

## (1) 간단 체인: 수도 질의

1. User Input: {"country": "France"} 
2. PromptTemplate (capital prompt) 
3. LLM (ChatOllama) 
4. StrOutputParser (AIMessage → str) 
5. Final Output: "Paris."

------------------------------------------------------------------------

## (2) 복합 체인: 정보 → 국가 추론 → 수도 반환

1. User Input: {"information": "...", "continent": "Europe"} 
2. RunnablePassthrough (입력 보존/구조 유지) 
3. country_chain = (country_prompt → LLM → StrOutputParser) 
4. {"country":"`<추론된 국가명>`{=html}"} 
5. capital_chain = (capital_prompt → LLM → StrOutputParser) 
6. Final Output: "`<수도>`{=html}

------------------------------------------------------------------------

# 핵심 요약

-   LCEL의 `|`는 "앞 출력 → 뒤 입력"을 연결하는 **파이프**다.
-   `PromptTemplate`, `LLM`, `OutputParser`는 모두 Runnable처럼
    동작하므로 서로 연결 가능하다.
-   `capital_chain = prompt | llm | parser`는 수동 호출 3단계를 한 줄로
    대체한다.
-   dict 파이프(`{...} | {...}`)를 쓰면 입력을 분기/병합하여 복잡한
    데이터 플로우를 만들 수 있다.
-   `RunnablePassthrough`는 "입력을 그대로 전달"하는 Runnable로, 복합
    체인 구성에 자주 쓰인다.
