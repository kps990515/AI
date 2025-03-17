## PyTorch 소개

### PyTorch란?
파이썬(Python)으로 딥러닝 모델을 만들고 학습(훈련)시키는 것을 도와주는 머신러닝 프레임워크입니다.

### PyTorch의 장점
- **자동 미분(Autograd)**: 딥러닝 학습에 필수적인 미분을 자동으로 수행합니다.
- **GPU 활용**: GPU를 사용하여 빠르게 대용량 데이터를 처리할 수 있습니다.
- **직관적 코딩**: 작성한 코드가 바로 실행되는 동적 구조로, 디버깅이 쉽습니다.

### 쉽게 이해하기
- **NumPy**: 숫자 데이터를 다루는 "엑셀" 같은 도구로, CPU만 활용 가능
- **PyTorch 텐서**: NumPy와 비슷한 형태로 GPU까지 활용하여 대규모 데이터 처리 가능

---

## PyTorch 핵심 기능

### 1. 텐서(Tensor)
- NumPy 배열과 유사한 구조의 다차원 데이터 객체
- GPU 연산 지원으로 빠른 속도 보장

#### 예시 코드
```python
import torch

# 1차원 텐서
# 1차원 텐서(벡터)를 생성
x = torch.tensor([1., 2., 3.])

# 2차원 텐서 (행렬)
# 텐서 y는 2개의 행과 2개의 열을 가지며, shape는 [2, 2]입니다. 행렬의 첫 번째 행은 [1.0, 2.0], 두 번째 행은 [3.0, 4.0]
y = torch.tensor([[1., 2.], [3., 4.]])

# 차원 추가
# unsqueeze: 텐서의 특정 위치(0)에 크기가 1인 차원을 추가하는 메서드
# y_expanded : tensor([[1., 2., 3.]])
y_expanded = x.unsqueeze(0)  # torch.Size([1, 3])
```

### 자동 미분(Autograd)
- 미분 연산을 자동화하여 신경망 학습을 돕는 기능
- 복잡한 수학적 계산을 파이토치가 대신 수행

#### 예시 코드
```python
# requires_grad=True : 텐서의 연산 기록을 자동으로 추적하여, 나중에 역전파(backward) 과정을 통해 미분(gradient)을 계산할 수 있도록 하는 설정
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x[0]**2 + x[1]**2  # y = x0^2 + x1^2

# 자동 미분을 통해 y를 x에 대해 미분한 결과를 계산
# 이때, 각 변수에 대한 기울기(gradient)가 x.grad에 저장
y.backward()

print(x.grad)  # 출력: tensor([4., 6.])
```

### PyTorch로 딥러닝 맛보기 (간단한 선형회귀)
```python
# 1. 데이터 준비
# 입력데이터
x_train = torch.tensor([[1.], [2.], [3.], [4.]])
# 목표데이터
y_train = torch.tensor([[3.], [5.], [7.], [9.]])

# 2. 모델 정의
# nn.Module: 모든 PyTorch 모델의 기본 클래스
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
# 입력 특성(feature) 수가 1, 출력 특성 수가 1인 간단한 선형 계층으로, 수학적으로는 y=Wx+b 형태의 연산을 수행

# 3. 모델에 입력 데이터를 넣었을 때 어떻게 계산을 수행할지 정의하는 함수
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
# nn.MSELoss(): 예측 값과 실제 값 사이의 차이를 제곱하여 평균을 내는 손실 함수
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# optim.SGD: Stochastic Gradient Descent (SGD) 옵티마이저를 사용하여 모델의 파라미터를 업데이트
# model.parameters()를 통해 모델의 학습 가능한 파라미터들을 전달하며, lr=0.01은 학습률(learning rate)을 의미

# 4. 학습 루프
for epoch in range(100): # 총 100번의 학습(에포크, epoch)을 수행
    # 모델의 forward 메서드가 호출되어, 현재 모델 파라미터로 x_train 데이터를 입력받아 예측 값을 계산
    pred = model(x_train) 
    # 예측 값 pred와 실제 값 y_train 사이의 오차(손실)를 계산
    loss = criterion(pred, y_train)
    # 매 학습 단계마다 기울기를 0으로 만들어 누적되지 않도록 합니다.
    optimizer.zero_grad()
    # 역전파(backpropagation)를 통해 손실 함수의 결과를 모델 파라미터에 대해 미분(기울기 계산)합니다.
    loss.backward()
    # 계산된 기울기를 바탕으로 모델의 파라미터를 업데이트
    # 이 과정에서 학습률(lr)에 따라 파라미터가 조정되며, 손실이 줄어들도록 최적화
    optimizer.step()
```

### 필수로 기억할 PyTorch 학습 흐름

| 단계      | 설명                                                          |
|------------|----------------------------------------------------|
| 데이터 준비 | 데이터 → 텐서 변환 및 DataLoader를 통한 로딩 |
| 모델 정의 | nn.Module 클래스 상속하여 정의 |
| 손실함수 | 예측값과 실제값 차이 계산 (MSE, CrossEntropy 등) |
| 옵티마이저 | 파라미터 업데이트 (SGD, Adam 등) |
| 학습 루프 | 순전파 → 손실 계산 → 역전파 → 파라미터 업데이트 |
| 평가 및 추론 | 테스트 데이터셋을 활용한 성능 평가 및 예측 |

### PyTorch 주요 흐름 (표 정리)
| 단계 | 설명 |
|------|------|
| 데이터 준비 | CSV, 이미지 등 데이터를 텐서로 변환. DataLoader로 batch 단위 불러오기 |
| 모델 정의 | nn.Module 상속하여 신경망 모델 작성 |
| 손실함수 | 예측 결과와 실제 값 비교하여 오차 측정 |
| 옵티마이저 | SGD, Adam 등을 이용하여 파라미터 업데이트 |
| 학습 루프 | forward → loss → backward → update |
| 평가 및 추론 | 학습된 모델 성능 평가 및 새로운 데이터 예측 |

