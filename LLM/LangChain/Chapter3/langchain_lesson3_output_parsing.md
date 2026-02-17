# 실습 주제

-   **한 줄 요약:** LangChain의 OutputParser와
    `with_structured_output()`을 이용해 LLM 응답을 **문자열** 또는
    **구조화된(Pydantic) 객체** 형태로 안정적으로 다루는 방법을
    학습한다.
-   **해결하려는 문제:**
    -   LLM의 기본 응답은 `AIMessage`로 오며, 텍스트 기반이기 때문에
        후속 로직(파싱/검증/저장/API 응답)에 바로 쓰기 어렵다.
    -   `StrOutputParser`로 "문자열만" 쉽게 추출하거나,
        `Pydantic(BaseModel)`을 이용해 **필드가 있는 데이터 구조로
        강제**하면, 백엔드 서비스에서 신뢰할 수 있는 타입/스키마 기반
        처리가 가능해진다.

------------------------------------------------------------------------

# 전체 구조 설명

이 노트북은 "LLM 출력 형식 제어"를 두 레벨로 다룹니다.

1)  **단순 문자열 출력**: `StrOutputParser`로 `AIMessage` → `str` 변환
2)  **구조화된 출력(Structured Output)**: Pydantic 모델을 스키마로
    제공하여 LLM 출력 → `CountryDetail` 객체로 받기

전체 흐름:

1.  `ChatOllama`로 LLM 초기화
2.  `PromptTemplate`로 입력 변수가 포함된 프롬프트 생성
3.  `llm.invoke()` 실행 결과가 `AIMessage`임을 확인
4.  `StrOutputParser`로 `AIMessage`를 `str`로 변환
5.  Pydantic 모델 정의 (`CountryDetail`)
6.  `llm.with_structured_output(CountryDetail)`로 구조화된 LLM 래핑
7.  JSON 지시 프롬프트로 `CountryDetail` 객체 생성 응답 받기
8.  `model_dump()`로 dict 변환 후 특정 필드 접근

------------------------------------------------------------------------

# LangChain 주요 개념 설명 (이 실습 기준)

## 1) LLM

-   **역할:** 프롬프트를 입력받아 답변 생성
-   **이 실습에서 사용:**

``` python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2:1b")
```

-   **입력 타입:** string / PromptValue / messages
-   **출력 타입:** 기본적으로 `AIMessage`

------------------------------------------------------------------------

## 2) PromptTemplate

-   **역할:** `{country}` 같은 변수를 포함한 템플릿을 정의하고, 실행 시
    실제 값으로 치환해 "최종 프롬프트"를 만든다.
-   **이 실습에서 사용:**

``` python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_template = PromptTemplate(
    template="What is the capital of {country}? Return the name of the city only",
    input_variables=["country"],
)
prompt = prompt_template.invoke({"country": "France"})
```

-   `prompt`는 LangChain의 `PromptValue` 형태로 만들어지며, 출력 시
    `text='...'`처럼 보일 수 있다.

------------------------------------------------------------------------

## 3) OutputParser

-   **역할:** LLM 출력(`AIMessage` 또는 텍스트)을 우리가 쓰기 좋은
    형태로 변환
-   **이 실습에서 사용:** `StrOutputParser`
-   **추가 언급:** `JsonOutputParser`(주석 처리되어 있음)

### StrOutputParser

-   `AIMessage`에서 `content`만 꺼내서 `str`로 반환

``` python
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
answer = output_parser.invoke(ai_message)  # "Paris"
```

------------------------------------------------------------------------

## 4) Runnable

-   **역할:** `.invoke()`로 실행 가능한 단위
-   `PromptTemplate`, `OutputParser`, `LLM` 모두 Runnable처럼 동작한다.
-   그래서 아래처럼 각각 `.invoke()`가 가능하다.

``` python
prompt_template.invoke(...)
llm.invoke(...)
output_parser.invoke(...)
```

> 4강 LCEL에서 `prompt | llm | parser`로 연결되는 기반 개념이다.

------------------------------------------------------------------------


# 코드 흐름 상세 분석

## Step 1. LLM 초기화

``` python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2:1b")
```

-   로컬 Ollama 모델을 LangChain ChatModel로 생성
-   이후 모든 실습은 이 `llm.invoke(...)`로 수행

------------------------------------------------------------------------

## Step 2. 문자열 출력 파서(StrOutputParser) 사용

### 2-1) PromptTemplate로 명시적 지시문 포함 프롬프트 만들기

``` python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_template = PromptTemplate(
    template="What is the capital of {country}? Return the name of the city only",
    input_variables=["country"],
)
prompt = prompt_template.invoke({"country": "France"})
print(prompt)
```

출력 예(노트북 출력):

    text='What is the capital of France? Return the name of the city only'

핵심 포인트:

-   `PromptTemplate.invoke()`는 "값이 치환된 프롬프트"를 만든다.
-   이 결과는 단순 문자열이 아니라 PromptValue 계열로 표현될 수 있어
    출력이 `text='...'`로 보인다.
-   하지만 `llm.invoke()`에 그대로 넘길 수 있다.

------------------------------------------------------------------------

### 2-2) LLM 호출 결과 확인 (`AIMessage`)

``` python
ai_message = llm.invoke(prompt_template.invoke({"country": "France"}))
print(ai_message)
```

-   `ai_message`는 `AIMessage` 타입
-   실제 답변 텍스트는 `ai_message.content`

------------------------------------------------------------------------

### 2-3) StrOutputParser로 문자열 변환

``` python
output_parser = StrOutputParser()
answer = output_parser.invoke(llm.invoke(prompt_template.invoke({"country": "France"})))
```

이 때 `answer`는 `"Paris"` 같은 **순수 문자열(str)** 이 된다.

왜 필요한가?

-   실무에서는 "LLM 응답 텍스트만" API 응답으로 보내거나
-   후속 파싱/정규화/DB 저장을 해야 한다.
-   매번 `ai_message.content`를 꺼내는 대신 파서를 두면 파이프라인화가
    쉬워진다.

------------------------------------------------------------------------

## Step 3. 응답 타입 확인

``` python
type(ai_message.content)  # str
answer                    # 'Paris'
```

핵심: - `AIMessage.content`는 이미 `str`이지만, - `AIMessage` 전체에는
metadata/usage 등 부가 정보가 있으므로 - "문자열만" 필요하면 파서로
정리하는 게 깔끔하다.

------------------------------------------------------------------------

## Step 4. 구조화된 출력(Structured Output) 사용

여기부터가 실무적으로 가장 강력한 파트입니다.

### 4-1) Pydantic 모델로 스키마 정의

``` python
from pydantic import BaseModel, Field

class CountryDetail(BaseModel):
    capital: str = Field(description="The capital of the country")
    population: int = Field(description="The population of the country")
    language: str = Field(description="The language of the country")
    currency: str = Field(description="The currency of the country")
```

의미: - LLM이 "자유 텍스트"가 아니라 -
`capital/population/language/currency` 필드를 갖는 구조로 답하도록
유도/강제한다. - 백엔드 입장에서는 DTO 스키마처럼 다루는 것과 동일하다.

------------------------------------------------------------------------

### 4-2) `with_structured_output()`로 LLM 래핑

``` python
structued_llm = llm.with_structured_output(CountryDetail)
```

핵심: - 이 순간부터 `structued_llm.invoke(...)`의 결과는 `AIMessage`가
아니라 - **`CountryDetail` 객체**가 된다.

즉: - 문자열 파싱이 아니라 - "스키마 기반 결과"를 바로 받는다.

------------------------------------------------------------------------

### 4-3) JSON을 요청하는 프롬프트 작성

``` python
country_detail_prompt = PromptTemplate(
    template="""Give following information about {country}:
    - Capital
    - Population
    - Language
    - Currency

    return it in JSON format. and return the JSON dictionry only 
    """,
    input_variables=["country"],
)
```

포인트: - 구조화 출력은 프롬프트에서 **출력 형식을 강하게 요구할수록
성공률이 올라간다.** - "JSON only" 같은 문구는 파싱 실패를 줄이는 대표
패턴

------------------------------------------------------------------------

### 4-4) 구조화된 응답 확인

``` python
json_ai_message = structued_llm.invoke(country_detail_prompt.invoke({"country": "France"}))
json_ai_message
```

출력 예:

    CountryDetail(capital='Paris', population=65275746, language='French', currency='Euro')

이제 결과는 `CountryDetail` 객체이므로, 아래처럼 바로 필드 접근이
가능하다.

------------------------------------------------------------------------

### 4-5) 특정 필드 접근

``` python
json_ai_message.model_dump()["capital"]
```

-   `model_dump()`는 Pydantic 객체 → dict 변환
-   또는 `json_ai_message.capital`처럼 바로 접근도 가능

------------------------------------------------------------------------

# 주의사항(실무 관점)

## 1) LLM이 "정확한 숫자(인구)"를 항상 맞추지는 않는다

-   `population` 같은 사실 데이터는 모델 환각 가능성이 있다.
-   실무에서는 **RAG(문서 기반)** 또는 **외부 API 기반**으로 사실을
    보강하는 게 일반적이다.

## 2) JSON only 지시가 약하면 파싱이 깨질 수 있다

-   모델이 설명 텍스트를 덧붙이면 파싱 실패
-   그래서 시스템 메시지/프롬프트에서 "오직 JSON만"을 강하게 요구하는
    패턴이 자주 사용된다.

------------------------------------------------------------------------

# 실행 흐름 다이어그램 (텍스트 기반)

## (1) StrOutputParser 기반 문자열 출력 흐름

1. User Input 
2. PromptTemplate (format: country 치환) 
3. LLM (ChatOllama) 
4. AIMessage 
5. StrOutputParser (AIMessage → str) 
6. Final Output (예:"Paris")

------------------------------------------------------------------------

## (2) Pydantic 기반 구조화 출력 흐름

1. User Input 
2. PromptTemplate (JSON 출력 지시 포함) 
3. LLM.with_structured_output(CountryDetail) 
4. CountryDetail 객체(Pydantic) 
5. Final Output (json_ai_message.capital 등으로 접근)

------------------------------------------------------------------------

# 핵심 요약

-   LLM의 기본 결과는 `AIMessage`이며, `content`에 텍스트가 들어있다.
-   `StrOutputParser`는 `AIMessage`를 깔끔한 `str`로 변환해준다.
-   `with_structured_output(PydanticModel)`을 쓰면 LLM 결과를 **스키마
    기반 객체**로 받을 수 있다.
-   구조화 출력은 "프롬프트에서 출력 형식을 강하게 요구할수록"
    안정적이다.
