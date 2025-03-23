# Python 개요 및 기본 문법 정리

## 1. 파이썬 개요

파이썬(Python)은 **1991년 귀도 반 로섬(Guido van Rossum)**이 만든 고급 프로그래밍 언어.

### 특징
- 문법이 간결하고 가독성이 좋음 (**들여쓰기를 문법의 일부로 사용**)
- 플랫폼 독립적 (Windows, macOS, Linux 등)
- 방대한 표준 라이브러리 제공

### 간단한 예제
```python
>>> print("Hello, Python!")
Hello, Python!
```

---

## 파이썬 설치 및 환경 설정

### 설치
- [Python 공식 사이트](https://www.python.org)에서 운영체제에 맞는 설치 파일 다운로드 후 설치
- 설치 시 **“Add Python to PATH” 옵션**(Windows)을 체크

### 가상환경 설정 (권장)
```bash
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate
```

---

## 파이썬 기본 문법

### 변수
```python
name = "Alice"
age = 25
print(name, age)  # Alice 25
```

### 자료형
- **숫자형**: 정수(int), 실수(float), 복소수(complex)
- **문자열**: str
- **불리언**: bool (`True` / `False`)

### 연산자
- 산술 연산자: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- 예시:
```python
a, b = 5, 2
print(a / b)   # 2.5
print(a // b)  # 2
print(a ** b)  # 25
```

---

## 문자열 활용

### 문자열 주요 메서드
- 길이: `len(str)`
- 대소문자 변환: `str.upper()`, `str.lower()`
- 치환: `str.replace(old, new)`
- 슬라이싱: `str[start:end:step]`

```python
str = "Hello Python"
print(len(str))                # 12
print(str.upper())            # HELLO PYTHON
print(str.lower())            # hello python
print(str.replace("Hello", "Hi"))  # Hi Python
```

### 포매팅 (f-string)
```python
name, age = "Alice", 25
print(f"My name is {name}, and I'm {age} years old.")
# My name is Alice, and I'm 25 years old.
```

---

## 변수와 연산자

### 변수와 할당
```python
x = 10
y = 20
result = x + y
print(result)  # 30
```

---

## 자료구조

### 리스트(list)
```python
numbers = [1, 2, 3]
numbers.append(4)
print(numbers)  # [1, 2, 3, 4]
```

### 튜플(tuple)
- 불변(immutable)
```python
my_tuple = (1, 2, 3)
```

### 딕셔너리(dictionary)
```python
person = {"name": "Bob", "age": 30}
person["age"] = 31
print(person)  # {"name": "Bob", "age": 31}
```

### 집합(set)
- 중복 제거 및 집합 연산 지원
```python
set_a = {1, 2, 3}
set_b = {3, 4, 5}
print(set_a | set_b)  # {1, 2, 3, 4, 5}
print(set_a & set_b)  # {3}
```

---

## 제어문

### 조건문(if-else)
```python
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

print(grade)  # B
```

### 반복문
#### while
```python
count = 0
while count < 3:
    print(count)
    count += 1
```

#### for
```python
for i in range(3):
    print(i)
# 0
# 1
# 2
```

#### break와 continue
- `break`: 즉시 반복문 종료
- `continue`: 현재 반복 건너뛰기
```python
for i in range(5):
    if i == 2:
        continue
    if i == 4:
        break
    print(i)
# 0, 1, 3
```

---

## 함수

### 기본 함수
```python
def greet(name):
    return f"Hello, {name}!"

print(greet("Python"))  # Hello, Python!
```

### 가변 인자(*args, **kwargs)
```python
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3))  # 6


def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Tom", age=20)
# name: Tom
# age: 20
```

---

## 클래스와 객체지향 프로그래밍

### 클래스 생성과 객체
```python
class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, my name is {self.name}.")

p1 = Person("Alice")
p1.greet()  # Hello, my name is Alice.
```

### 상속
```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def bark(self):
        print("Woof!")

dog = Dog("Buddy")
dog.bark()  # Woof!
```

---

## 예외 처리
```python
try:
    x = int("abc")
except ValueError:
    print("숫자로 변환 불가능")
finally:
    print("예외처리 완료")
```

---

## 유용한 표현

### 리스트 컴프리헨션
```python
numbers = [1, 2, 3, 4, 5]
squares = [x*x for x in numbers]
print(squares)  # [1, 4, 9, 16, 25]
```

### lambda 표현식
```python
nums = [1, 2, 3, 4, 5]
even_nums = list(filter(lambda x: x % 2 == 0, numbers))
print(even_nums)  # [2, 4]
```

---

## 코드 스타일 가이드
- 함수: `my_function`
- 클래스: `MyClass`
- 코드 줄바꿈과 주석 활용

---

**작성된 코드 및 내용을 지속적으로 연습하여 숙련도를 높이는 것을 권장합니다.**

