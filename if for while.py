# weather = input("오늘 날씨는 어때요? ")
# if weather =="비" or weather == "눈":
#     print("우산을 챙기세요")
# elif weather =="미세먼지":
#     print("마스크를 챙기세요")
# else:
#     print("준비물 필요 없어요.")

# temp = int(input("기온은 어때요? "))
# if 30 <= temp :
#     print("너무 더워요. 나가지 마세요")
# elif 10 <= temp and temp < 30:
#     print("괜찮은 날씨에요")
# elif 0<= temp <10:
#     print("외투를 챙기세요")
# else:
#     print("너무 추워요. 나가지 마세요.")


# starbucks = ["아이언맨", "토르", "아이엠그루트"]
# for customer in starbucks:
#     print("{0}, 커피가 준비되었습니다.".format(customer))

# #while
# customer = "토르"
# index = 5
# while index >=1 :
#     print("{0}, 커피가 준비되었습니다. {1}번 남았어요.".format(customer , index))
#     index -= 1 
#     if index == 0:
#         print("커피는 폐기되었습니다.")

# customer = "아이언맨"
# index = 1
# while index >=1 :
#     print("{0}, 커피가 준비되었습니다. 호출 {1}회".format(customer , index))
#     index += 1


# customer = "토르"
# person = "Unknown"

# while person != customer :
#     print("{0}, 커피가 준비되었습니다.".format(customer))
#     person = input("이름이 어떻게 되세요? ")



# absent = [2 , 5]
# no_book = [7]
# for student in range(1,11) : 
#     if student in absent:
#         continue
#     elif student in no_book :
#         print("오늘 수업 끝. {0}은 교무실로 따라와".format(student))
#         break
#     print("{0}, 책을 읽어봐.".format(student))


#퀴즈5 내가 푼 답
from random import*
time = randrange(5,51)
num = 0
for psg in range(1,51):
    if time >= 5 and time <=15 :
        ride = "O"
        num += 1
    else:
        ride = " "
    print("[{0}] {1}번째 손님 (소요시간 : {2}분)".format(ride,psg,time ))
    
    time = randrange(5,51)
print("총 탑승 승객 : {0}분".format(num))
