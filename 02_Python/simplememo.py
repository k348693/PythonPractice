
import os
os.makedirs("save_dir", exist_ok = True)  # 디렉토리 생성

print("작성한 내용을 저장할 파일명을 입력하세요.")
file_name = input("파일명: ")
#저장할 내용을 모아놓을 리스트 
txt_list = []
while True:
    txt = input(">>")
    if txt == "!q" : 
        break
    txt_list.append(txt + "\n")
# print(os.path.join("save_dir",file_name))
with open(os.path.join("save_dir",file_name), "wt" ) as fw:
    fw.writelines(txt_list)
    txt_list[-1] = txt[-1][:-1]  # 마지막 문장에 붙인 엔터 제거
