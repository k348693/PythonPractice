# jupyterlab 기능
# cell 내용을 파일경로에 저장. (실행하지는 않는다.)
# cell 첫줄에 작성

# 인사말 관련 모듈.
__version__ = 1.0

def print_greeting(name):
    print(f"{name}님 안녕하세요.")

def print_welcome(name):
    print(f"{name}님 환영합니다.")

def get_greeting(name):
    print(f"{name}님 안녕하세요.")

class Hello:
    def __init__(self, name):
        self.name = name

    def kor_greet(self):
        return f"{self.name}님 안녕하세요."

    def eng_greet(self):
        return f"Hello {self.name}~!"

print("SDFDF)")
print(__name__)