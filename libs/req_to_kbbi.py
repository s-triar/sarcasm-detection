import requests
from bs4 import BeautifulSoup
import re

session = requests.Session()
URL = "https://kbbi.kemdikbud.go.id/entri/"


URL_LOGIN = "https://kbbi.kemdikbud.go.id/Account/Login"

page_login = session.get(URL_LOGIN)
html = page_login.text
# print(html)
soup = BeautifulSoup(html, 'html.parser')
cls_res_area = soup.find_all('input')
# print(cls_res_area)
# y = "<input name=\"__RequestVerificationToken\" type=\"hidden\" value=\"T_Rw0Z8XgqK02KhZ5Xrt3ZQztf3uUZXhN1m4Zvc6HYos_Ywa2flLempfu39jERlVRbd4HLqiZMRziX_cJU-gVNtVvt2OaQ5IX4v-wjA8R5A1\"/>"
req_token = re.findall("value=\"(.+)\"",str(cls_res_area[0]))
# print(req_token[0])

data = {'__RequestVerificationToken': req_token[0],'Posel':'emailnama@gmail.com','KataSandi':'pass','IngatSaya':True}
x = session.post(URL_LOGIN, data = data)
# print(x)
sess = session.cookies.get_dict()
print(sess)

def check_with_kkbi(word):
      r = session.get(url = URL+word)
      txt = r.text
    #   print(txt)
      beku = txt.find('Akun Dibekukan')
    #   print("beku", beku)
      if(beku!=-1):
          raise ValueError("Akun KBBI Dibekukan")
      max_pencarian = txt.find('Pencarian Anda telah mencapai batas maksimum dalam sehari')
      if(max_pencarian!=-1):
          raise ValueError('Pencarian Anda telah mencapai batas maksimum dalam sehari')
      failed = txt.find('Terjadi Kesalahan')
      if(failed!=-1):
          raise ValueError('Terjadi Kesalahan')
      cls_res_area = txt.find('Entri tidak ditemukan.')
      return cls_res_area==-1
  
# f = check_with_kkbi('aku')
# print(f)