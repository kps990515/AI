# Numpy 기본 개념과 활용
 - Python에서 수치 계산과 데이터 분석을 위한 핵심 라이브러리
   - 다차원 배열
   - 벡터화 연산
   - 수학, 통계, 선형대수 지원
 
## 0. Numpy 시작하기

먼저 Numpy를 사용하기 위해 아래와 같이 import 합니다.

```python
import numpy as np
```

---

## 1. Array 생성 및 기본 속성

Numpy의 핵심은 array입니다. 일반 Python list와 달리 모든 원소의 타입이 동일하며, 여러 수학 및 통계 연산을 효율적으로 수행할 수 있습니다.

### 1.1 Python list → Numpy array
```python
arr = np.array([1, 2, 3, 4, 5])
print(arr, type(arr))
```
**출력:**
```
[1 2 3 4 5] <class 'numpy.ndarray'>
```

- `ndim`: 차원의 수
- `size`: 원소의 총 개수
- `shape`: 각 차원의 크기

```python
print(arr.ndim, arr.size, arr.shape)
```
**출력:**
```
1 5 (5,)
```

### 1.2 2차원 Array 생성
행렬과 같이 생성할 수 있습니다.
```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d)
print(arr_2d.ndim, arr_2d.shape, arr_2d.size)
```
**출력:**
```
[[1 2 3]
 [4 5 6]]
2 (2, 3) 6
```

### 1.3 데이터 타입(dtype)
모든 원소가 동일한 타입을 갖습니다.
```python
arr_float = np.array([1., 2, 3, 4, 5])
print(arr_float.dtype)
```
**출력:**
```
float64
```

### 1.4 상수값으로 채워진 Array 생성
```python
zeros = np.zeros(5)
print(zeros)
print(zeros.shape)
```
**출력:**
```
[0. 0. 0. 0. 0.]
(5,)
```

다차원 배열 생성:
```python
zeros_nd = np.zeros((32, 32, 3))
print(zeros_nd.shape)
```
**출력:**
```
(32, 32, 3)
```

### 1.4 연속된 수열 생성하기
- `np.arange`: 간격을 지정하여 생성
```python
range1 = np.arange(5)
range4 = np.arange(0, 10, 0.5)
print(range1, range4)
```

- `np.linspace`: 개수를 지정하여 균등 간격 생성
```python
range1 = np.linspace(0, 4, num=5)
print(range1)
```

### 1.4 Random Array 생성
```python
arr = np.random.rand(2, 3, 4)
print(arr)
```

## 2. Array의 Shape와 차원 조작

### 2.1 reshape
원소 개수를 유지하면서 형태를 변경합니다.
```python
arr = np.array([1, 2, 3, 4, 5, 6])
arr_reshaped = arr.reshape(2, 3)
print(arr_reshaped)
```

### 2.2 차원 확장
`np.newaxis` 또는 `None`을 사용하여 차원을 추가합니다.
```python
arr = np.array([1, 2, 3], [4, 5, 6]) // (2,3)
arr_expanded = arr[:, None] // : 모든행에, None: 새로운 차원 추가
print(arr_expanded.shape) // (2,1,3)
```

### 2.3 concatenate 및 stack
- `np.concatenate`: 지정한 축(axis)을 따라 배열을 연결합니다.
```python
arr1 = np.zeros((2, 3, 5))
arr2 = np.zeros((2, 1, 5))
arr3 = np.zeros((2, 4, 5))
arr_concat = np.concatenate([arr1, arr2, arr3], axis=1) // 두번째 축을 기준으로 합쳐짐
print(arr.shape) // 2,8,5
```

- `np.stack`: 새로운 축을 만들어 배열을 쌓습니다.
```python
arr1 = np.zeros((2, 3))
arr_stacked = np.stack([arr1, arr1, arr1], axis=0) //맨앞에 새로운 차원을 넣는데 arr1을 3번 넣음
print(arr_stacked.shape) // (3,2,3) 3:axis=0 3개, 2: 기존행, 3: 기존열
```

### 2.4 transpose 및 flatten
- `transpose`: 배열의 차원을 전치합니다.
```python
arr = np.random.rand(2, 3, 4)
arr_transposed = arr.transpose(1, 0, 2) // 0,1축을 바꾸고 2축은 그대로 유지
print(arr_transposed.shape) // (3,2,4)
```

- `flatten`: 1차원으로 만듭니다.
```python
arr_flat = arr.flatten()
print(arr_flat.shape)
```

## 3. Broadcasting 개념

### Broadcasting이 가능한 조건
- 두 배열의 차원을 뒤에서부터 비교할 때, 해당 차원의 크기가 같거나, 둘 중 하나가 1인 경우 가능합니다.
- 예시: `(2, 3)`과 `(3,)`의 경우 (3,)가 (1, 3)으로 확장되어 연산이 가능합니다.

### Broadcasting이 불가능한 조건
- 비교하는 차원 중 하나도 일치하지 않고, 한쪽도 1이 아닌 경우는 불가능합니다.
- 예시: `(2,3)`과 `(3,2)`는 불가능합니다.

### Broadcasting 예시
```python
arr1 = np.random.rand(2, 3)
arr2 = np.array([1, 2, 3])
result = arr1 + arr2
print(result.shape)
```
**출력:**
```
(2, 3)
```

배치 행렬 곱셈에서도 적용 가능합니다.
```python
arr1 = np.random.rand(4, 1, 2, 3)
arr2 = np.random.rand(1, 5, 3, 4)
result = arr1 @ arr2
print(result.shape)
```
**출력:**
```
(4, 5, 2, 4)
```

---

## 요약 정리

- **Array 생성**: `array`, `zeros`, `ones`, `arange`, `linspace`, `random`
- **Shape 조작**: `reshape`, `newaxis/None`, `expand_dims`, `concatenate`, `stack`, `transpose`, `flatten`
- **연산**: `max`, `min`, `sum`, `mean`, `std`, `unique`, `argsort`
- **행렬 연산**: `@`, `matmul`
- **Broadcasting**: 차원 일치 혹은 1인 경우 자동 확장하여 연산 가능

이 문서를 바탕으로 다양한 연습을 진행하며 Numpy를 완벽히 익혀봅시다.

