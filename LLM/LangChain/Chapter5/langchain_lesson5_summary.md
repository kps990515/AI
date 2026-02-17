# 실습 주제

-   **한 줄 요약:** Temperature, PromptTemplate, ChatPromptTemplate,
    LCEL 체인 연결을 활용하여 "국가 → 대표 음식 → 레시피 생성"까지
    자동으로 수행하는 LangChain 파이프라인을 구현한다.
-   **해결하려는 문제:**
    -   LLM을 단일 질문 응답이 아니라, 여러 단계의 추론 및 작업을
        자동으로 연결된 파이프라인으로 구성하고 싶다.
    -   또한 Temperature, 출력 형식 명시, SystemMessage 활용 등의
        프롬프트 엔지니어링 기법을 사용하여 **일관성 있고 안정적인
        결과를 얻는 방법**을 학습한다.

------------------------------------------------------------------------

# 전체 구조 설명

이 강의는 지금까지 배운 모든 개념을 하나의 실제 파이프라인으로 통합하는
것이 목표입니다.

전체 흐름:

1.  Temperature=0으로 설정된 LLM 생성
2.  PromptTemplate을 이용한 대표 음식 추천 체인 생성 (food_chain)
3.  ChatPromptTemplate을 이용한 레시피 생성 체인 구성 (recipe_chain)
4.  LCEL dict 파이프를 사용해 두 체인을 연결 (final_chain)
5.  최종 입력(country) → 음식 추천 → 레시피 생성 자동 수행

------------------------------------------------------------------------

# LangChain 주요 개념 설명

## 1. LLM (Temperature 설정)

``` python
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
)
```

### Temperature란?

LLM의 "랜덤성"을 제어하는 파라미터

  Temperature   특징
  ------------- -------------------------------------
  0             가장 deterministic (항상 비슷한 답)
  0.3           약간 다양성
  0.7           일반적인 기본값
  1.0           매우 창의적
  \>1           매우 랜덤

### 실무 권장값

  용도                   Temperature
  ---------------------- -------------
  QA / Agent / Backend   0 \~ 0.2
  일반 Chat              0.5 \~ 0.7
  창의적 생성            0.7 \~ 1.0

------------------------------------------------------------------------

## 2. PromptTemplate

대표 음식 찾기 Prompt

``` python
food_prompt = PromptTemplate(
    template='''what is one of the most popular food in {country}? 
Please return the name of the food only''',
    input_variables=['country']
)
```

체인 구성

``` python
food_chain = food_prompt | llm | StrOutputParser()
```

입력

    {'country': 'South Korea'}

출력

    "Bibimbap"

------------------------------------------------------------------------

## 3. ChatPromptTemplate + SystemMessage

레시피 생성 Prompt

``` python
recipe_prompt = ChatPromptTemplate.from_messages([
    ('system', '''Provide the recipe...
Please return numbered list only'''),
    ('human', 'Can you give me the recipe for making {food}?')
])
```

체인

``` python
recipe_chain = recipe_prompt | llm | StrOutputParser()
```

역할:

SystemMessage → 출력 형식 강제

HumanMessage → 사용자 요청

------------------------------------------------------------------------

## 4. LCEL Chain 연결 (핵심)

최종 체인

``` python
final_chain = {'food': food_chain} | recipe_chain
```

의미:

1.  입력 country → food_chain 실행 → food 반환
2.  반환된 food를 recipe_chain 입력으로 전달
3.  레시피 생성

------------------------------------------------------------------------

# 코드 흐름 상세 분석

## Step 1. food_chain

구조

    country
     ↓
    PromptTemplate
     ↓
    LLM
     ↓
    StrOutputParser
     ↓
    food name

예:

    France → Croissant

------------------------------------------------------------------------

## Step 2. recipe_chain

구조

    food
     ↓
    ChatPromptTemplate
     ↓
    LLM
     ↓
    StrOutputParser
     ↓
    recipe

------------------------------------------------------------------------

## Step 3. final_chain

구조

    country
     ↓
    food_chain
     ↓
    food
     ↓
    recipe_chain
     ↓
    recipe

------------------------------------------------------------------------

# 실행 흐름 다이어그램

    User Input 
     ↓ 
    food_chain 
     ↓ 
    food name 
     ↓ 
    recipe_chain 
     ↓ 
    recipe 
     ↓ 
    Final

------------------------------------------------------------------------

# Prompt Engineering 핵심 팁

## Tip 1. Temperature 사용

``` python
temperature=0
```

효과:

-   일관된 응답

------------------------------------------------------------------------

## Tip 2. 출력 형식 명시

Bad

    What is popular food in Korea?

Good

    Return the name of the food only

------------------------------------------------------------------------

## Tip 3. SystemMessage 사용

    Return numbered list only

효과:

출력 형식 제어 가능

------------------------------------------------------------------------

## Tip 4. Chain 연결

``` python
{'food': food_chain} | recipe_chain
```

효과:

자동 파이프라인

------------------------------------------------------------------------

# LCEL 전체 파이프라인 구조

    User Input (country)
     ↓
    food_prompt
     ↓
    LLM
     ↓
    food name
     ↓
    recipe_prompt
     ↓
    LLM
     ↓
    recipe

------------------------------------------------------------------------

# 핵심 요약

이번 강의는 지금까지 배운 모든 개념을 통합한 실전 예제이다.

사용된 핵심 기술:

-   PromptTemplate
-   ChatPromptTemplate
-   SystemMessage
-   StrOutputParser
-   LCEL chain 연결
-   Temperature 제어

최종 결과:

하나의 invoke 호출로 여러 단계 작업 수행 가능

    final_chain.invoke({'country': 'France'})
