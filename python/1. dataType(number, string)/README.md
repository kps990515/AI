## 숫자
```python
// 정수형
>>> a = 123
>>> a = -178
>>> a = 0

// 실수형
>>> a = 1.2
>>> a = -3.45
>>> a = 4.24E10
>>> a = 4.24e-10

// 8진수
a = 0o177

// 16진수
a = 0x8ff

// x의 y제곱
>>> a = 3
>>> b = 4
>>> a ** b

// 나누기
>>> 7 / 4
1.75

// 나누기 몫 구하기
>>> 7 // 4
1
```

## 문자열
```python
// 4가지 방법
"Python's favorite food is perl"
'"Python is very easy." he says.'

multiline='''
... Life is too short
... You need python
... '''

multiline="""
... Life is too short
... You need python
... """

// 문자열 곱하기
>>> a = "python"
>>> a * 2
'pythonpython'

// 문자열 길이 구하기
>>> a = "Life is too short"
>>> len(a)
17

// 문자열 인덱싱
>>> a = "Life is too short, You need Python"
>>> a[3]
'e'

>>> a = "Life is too short, You need Python"
>>> a[0:4]
'Life'
>>> a[19:]
'You need Python'
>>> a[:17]
'Life is too short'
>>> a[19:-7]
'You need'

>>> a = "20230331Rainy"
>>> date = a[:8]
>>> weather = a[8:]
>>> date
'20230331'
>>> weather
'Rainy'

// 문자열 바꾸기
>>> a = "Pithon"
>>> a[1]
'i'
>>> a[1] = 'y' // 에러남(문자열은 immutable)

>>> a = "Pithon"
>>> a[:1]
'P'
>>> a[2:]
'thon'
>>> a[:1] + 'y' + a[2:]
'Python'

// 문자열 포맷팅
// 숫자
>>> "I eat %d apples." % 3
'I eat 3 apples.'

// 문자열
>>> "I eat %s apples." % "five"
'I eat five apples.'

// 변수
>>> number = 3
>>> "I eat %d apples." % number
'I eat 3 apples.'

// 2개이상
>>> number = 10
>>> day = "three"
>>> "I ate %d apples. so I was sick for %s days." % (number, day)
'I ate 10 apples. so I was sick for three days.'

// 이름으로 넣기
>>> "I ate {number} apples. so I was sick for {day} days.".format(number=10, day=3)
'I ate 10 apples. so I was sick for 3 days.'
>>> d = {'name':'홍길동', 'age':30}
>>> f'나의 이름은 {d["name"]}입니다. 나이는 {d["age"]}입니다.'
'나의 이름은 홍길동입니다. 나이는 30입니다.'

// f 문자열 포매팅
>>> name = '홍길동'
>>> age = 30
>>> f'나의 이름은 {name}입니다. 나이는 {age}입니다.'
'나의 이름은 홍길동입니다. 나이는 30입니다.'
>>> age = 30
>>> f'나는 내년이면 {age + 1}살이 된다.'
'나는 내년이면 31살이 된다.'



// 정렬과 공백
>>> "%10s" % "hi"
'        hi'
>>> "%-10sjane." % 'hi'
'hi        jane.'

// 소수점
>>> "%0.4f" % 3.42134234
'3.4213'
>>> "%10.4f" % 3.42134234
'    3.4213'

// count
>>> a = "hobby"
>>> a.count('b')
2

// find
>>> a = "Python is the best choice"
>>> a.find('b')
14
>>> a.find('k')
-1

// index
>>> a = "Life is too short"
>>> a.index('t')
8
>>> a.index('k')
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
ValueError: substring not found

// join
>>> ",".join('abcd')
'a,b,c,d'

// replace
>>> a = "Life is too short"
>>> a.replace("Life", "Your leg")
'Your leg is too short'

// split
>>> a = "Life is too short"
>>> a.split()
['Life', 'is', 'too', 'short']
>>> b = "a:b:c:d"
>>> b.split(':')
['a', 'b', 'c', 'd']
```
