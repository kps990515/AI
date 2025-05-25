# 데이터불러오기

## 기본전제
[기본 데이터]
- 학습데이터 : onenavi_train.csv(7월 20일에서 24일 사이의 수도권 3~15km 주행데이터)
- 평가데이터 : onenavi_evaluation.csv(7월 27일에서 31일 사이의 수도권 3~15km 주행데이터)

[추가 데이터]
- 주소(시군구)데이터 : onenavi_pnu.csv(주행데이터를 기준으로 출발지의 주소 정보, key : RID)
- 신호등(갯수)데이터 : onenavi_signal.csv(주행데이터를 기준으로 경로의 신호등 갯수, key : RID)
- 날씨데이터 : onenavi_weather.csv(주행데이터를 기준으로 해당 일자의 수도권 날씨 정보, key : RID)

## 불러오기 
```python
import pandas as pd  # pandas 라이브러리 불러오기

# 1) 학습(train) / 평가(evaluation) 데이터 읽기
# sep="|" : 파일이 파이프(|)로 구분되어 있을 때 지정
df_train = pd.read_csv("onenavi_train.csv", sep="|")       # 7/20–24 주행 데이터
df_eval  = pd.read_csv("onenavi_evaluation.csv", sep="|")  # 7/27–31 주행 데이터

# 2) 각 데이터의 크기(행×열) 출력
print("학습 데이터 크기:", df_train.shape)
print("평가 데이터 크기:", df_eval.shape)

# 3) 학습 + 평가 데이터 합치기
# ignore_index=True : 합친 후 기존 인덱스 무시하고 0부터 순서대로 재부여
df_total = pd.concat([df_train, df_eval], ignore_index=True)
print("합친 전체 데이터 크기:", df_total.shape)

# 4) 추가 데이터(주소·신호등) 읽기
df_pnu    = pd.read_csv("onenavi_pnu.csv",    sep="|")  # 출발지 시도/시군구 정보
df_signal = pd.read_csv("onenavi_signal.csv", sep="|")  # 경로 상 신호등 개수

# 5) RID 기준으로 Inner Join(교집합) 병합
# on="RID" : 공통 키로 'RID' 사용
# how="inner" (기본값) : 양쪽에 모두 존재하는 RID만 결과에 포함
df_total = pd.merge(df_total, df_pnu,    on="RID", how="inner")
df_total = pd.merge(df_total, df_signal, on="RID", how="inner")

# 6) 최종 결과 확인
print("최종 병합된 데이터 크기:", df_total.shape)
display(df_total.head())  # 처음 다섯 행을 확인해 봅니다
```

### Pandas 라이브러리를 활용해서 'onenavi_train.csv'파일을 'df' 변수에 저장하고 그 Shape을 확인
```python
import pandas as pd

df = pd.read_csv("onenavi_train.csv", sep="|")
print(df.shape)
```

### onenavi_train.csv'파일과 'onenavi_evaluation.csv'를 'df_total' 변수에 저장하고 그 Shape을 확인
```python
import pandas as pd

df_train = pd.read_csv("onenavi_train.csv",sep="|")
df_eval = pd.read_csv("onenavi_evaluation.csv",sep="|")

df_total = pd.concat([df_train,df_eval],ignore_index=True)
# ignore_index=True : 데이터 합치고 새로운 인덱스 부여 
```

### 지역정보와 신호등정보를 'df_total'와 합치기
```python
import pandas as pd

df_pnu = pd.read_csv("onenavi_pnu.csv",sep="|") # 주소(시도/시군구 정보)
df_signal = pd.read_csv("onenavi_signal.csv",sep="|") # 경로의 신호등 갯수

# RID 기준으로 Inner Join
df_total=pd.merge(df_total,df_pnu , on="RID")
df_total=pd.merge(df_total,df_signal , on="RID")

df_total
```

