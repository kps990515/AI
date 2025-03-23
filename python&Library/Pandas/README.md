# 📚 Pandas 핵심 개념 정리

## 🔖 Pandas란?
**Pandas**는 Python에서 데이터를 손쉽게 조작하고 분석할 수 있도록 돕는 라이브러리입니다. 엑셀, SQL과 같은 구조화된 데이터를 다루기에 유용하며, 데이터 정제, 분석, 시각화 작업에 다양하게 활용됩니다.

## 🗃️ 주요 데이터 구조

### 1. Series
- **정의**: 1차원 구조로, 각 데이터는 인덱스와 연결됩니다.
- **비유**: 엑셀의 단일 열처럼 사용됩니다.

**예시 코드:**
```python
import pandas as pd
data = pd.Series([10, 20, 30, 40])
print(data)
```

### 2. DataFrame
- **정의**: 행과 열이 있는 2차원 테이블 형식의 데이터 구조입니다.
- **비유**: 엑셀 시트나 데이터베이스 테이블과 유사합니다.

**예시 코드:**
```python
import pandas as pd

# 데이터프레임 생성
df = pd.DataFrame({
    "이름": ["철수", "영희", "민수"],
    "나이": [20, 22, 23]
})

print(df)
```

## 📌 Pandas 주요 기능
- **데이터 로드 및 저장**: CSV, Excel 등 다양한 형식 지원
- **데이터 필터링**: 조건에 따라 필요한 데이터 선택 가능
- **정렬**: 특정 열 기준으로 데이터를 오름차순/내림차순으로 정렬
- **그룹화**: 데이터를 그룹 지어 통계(평균, 합계 등) 계산
- **시각화**: Matplotlib과 연계하여 그래프 표현 가능
- **결측치 처리**: 결측값을 찾아 제거하거나 특정 값으로 대체 가능

## 📋 자주 사용되는 함수 목록 및 설명

- **pd.Series()**: 1차원 배열 데이터 구조 생성
- **pd.DataFrame()**: 2차원 테이블 데이터 구조 생성
- **df.head()**: 데이터프레임의 처음 몇 개(기본 5개)의 행 출력
- **df.info()**: 데이터프레임의 기본 정보 출력(데이터 타입, 결측치 등)
- **df.describe()**: 숫자형 데이터의 통계 요약 제공(평균, 표준편차 등)
- **df.sort_values(by=...)**: 지정된 열 기준 데이터 정렬
- **df.groupby(...).mean()**: 특정 열로 그룹화 후 각 그룹 평균 계산
- **df["컬럼명"].value_counts()**: 특정 열의 값 빈도수 계산
- **df.isnull().sum()**: 각 열의 결측치 개수 확인
- **df.fillna(값)**: 결측치를 특정 값으로 대체
- **pd.read_csv()**: CSV 파일을 읽어 데이터프레임으로 변환
- **pd.read_excel()**: Excel 파일을 읽어 데이터프레임으로 변환
- **df.to_csv()**: 데이터프레임을 CSV 파일로 저장
- **df.to_excel()**: 데이터프레임을 Excel 파일로 저장

---

# 🚀 단계별 학습 가이드

### 1️⃣ Pandas 설치 및 임포트
```bash
pip install pandas
```
```python
import pandas as pd
```

### 2️⃣ Series 생성 및 조작
```python
# Series 생성
data = pd.Series([10, 20, 30, 40])
print(data)

# 인덱싱 및 슬라이싱
print(data[0])  # 첫 번째 요소
print(data[:2])  # 앞 두 요소
```

### 3️⃣ DataFrame 생성 및 조작
```python
# DataFrame 생성
df = pd.DataFrame({
    "이름": ["철수", "영희", "민수"],
    "나이": [23, 25, 22]
})

# 열 추가
df["성별"] = ["남", "여", "남"]

# 열 삭제
df.drop(columns=["성별"], inplace=True)
```

### 4️⃣ 데이터 로드 및 저장
```python
# CSV 파일 읽기
df = pd.read_csv("data.csv")

# CSV 파일 저장
df.to_csv("new_data.csv", index=False)
```
```python
# Excel 파일 읽기/저장
df = pd.read_excel("data.xlsx")
df.to_excel("new_data.xlsx", index=False)
```

### 5️⃣ 데이터 필터링 및 정렬
```python
# 필터링
df_filtered = df[df["나이"] > 21]

# 정렬
df_sorted = df.sort_values(by="나이", ascending=False)
```

### 6️⃣ 데이터 그룹화
```python
df_grouped = df.groupby("성별").mean()
```

### 7️⃣ 데이터 시각화
```python
import matplotlib.pyplot as plt

# 히스토그램 예시
plt.hist(df["나이"], bins=5)
plt.xlabel("나이")
plt.ylabel("빈도수")
plt.title("나이 분포")
plt.show()
```

### 8️⃣ 결측치 처리
```python
# 결측치 확인
print(df.isnull().sum())

# 결측치 채우기
df_filled = df.fillna(0)
```

## 🌟 초보자를 위한 쉬운 설명

### 🔹 Series는 무엇인가요?
한 가지 종류의 데이터만 담는 "단일 열"이라고 생각하면 됩니다. 예를 들어, 학생들의 나이만 담은 열입니다.

### 🔹 DataFrame은 무엇인가요?
엑셀의 스프레드시트처럼 여러 정보를 담는 "테이블"입니다.

### 🔹 쉽게 배우는 팁!
- **단계별 학습:** 작은 데이터부터 다뤄보며 익숙해지세요.
- **실제 데이터를 활용**해서 실습하면 이해하기 쉽습니다. 예를 들어 학생 성적표 데이터를 활용하면 자연스럽게 익숙해질 수 있습니다.

이 가이드를 따라 차근차근 실습하면 누구나 쉽게 Pandas를 활용할 수 있습니다.

