# LLM 원리와 작동 방식

## Transfer Learning
- data가 부족하거나 data가 충분해도 훨씬 빠르게 generalization이 잘되는 모델을 학습하는 방법
- 구성요소
  - Pre-trained 모델: 충분한 data로 학습하여 자연어 이해 능력이 좋은 모델
  - Fine-tuning: pre-trained 모델을 가지고 우리가 목표로 하는 자연어 처리 문제를 푸는 모델을 학습하는 방법

## BERT
- 트랜스포머 기반의 양방향 인코더 표현 모델
- un-supervised 방식으로 pre-trained 모델을 학습하는 방식
- 입력은 자연어(text를 그대로 긁어와서 사용)

### 사전 학습(Pre-training) 방식
1. Masked Language Modeling (MLM)
   - 입력 문장 중 일부 단어를 마스킹(mask)하고, 모델이 이 단어를 예측하도록 학습
   - 문장의 일부분을 지웠을 때, 나머지 부분들을 보고 맞추는 식으로 모델을 학습하면 자연어를 더 잘 이해할 수 있을 것
   - 동작방식
     - 특정 token mask
     - Mask한 token 예측
2. Next sentence prediction (NSP)
   - text가 두 문장으로 이루어져있을 때, 두 문장이 실제로 이어진 문장인지 아니면 별개의 문장인지 맞추는 식으로 학습

### 양방향 문맥 이해
### NLP 전 분야에 걸쳐 미세 조정 

## DistilBERT
- knowledge distillation(증류)이라는 방식을 활용하여 BERT와 비슷한 성능을 내지만 더 빠르고 가볍게 만든 pre-trained Transformer 모델
- 동작방식
  - Teacher, student 모델 선정 : teacher 모델은 성능이 좋은 pre-trained 모델, student 모델은 학습하고자 하는 작은 모델
  - Soft label 생성: teacher모델이 Mask한 token에 대한 확률분포를 student모델에 전달(사과 60%, 배 30%, 오렌지 10%)
  - Student 모델 학습

## GPT(Generative Pre-trained Transformer)
- 동작방식
   - next token prediction 
     - 기존의 MLM loss는 문장 중간의 token을 masking하여 맞추는 방식으로 학습
     - next token prediction은 중간이 아닌, 마지막 token을 masking하고 맞추는 방식으로 학습
   - Text generative model
     - 어떤 문장을 입력으로 넣습니다. 그러면 next word prediction과 똑같은 과정으로 주어진 문장 다음 token을 생성
     - 생성된 token을 주어진 문장에 concat합니다. 그리고 다시 GPT에 입력으로 넣어 token을 생성합니다. 이 과정을 반복
     - BERT의 [SEP] 나 GPT의 <|endoftext|>와 같은 문장의 끝을 의미하는 token이 나올 때까지 생성
   - Large language model (LLM)
   - Few-shot learning
     - “몇 개만”의 예시를 보여주고 모델이 새 작업을 수행하게 하는 방법
     - 데이터 준비 비용을 대폭 줄이면서도 강력한 사전 학습 모델을 다양한 태스크에 빠르게 적용
     - 예시 입력(입력 -> 출력 예시)
     - 새로운 입력 제시(이런 입력에는 이런 출력을 해야 한다를 유추)
     - 모델 예측



