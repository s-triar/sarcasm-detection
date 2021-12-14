import requests
from bs4 import BeautifulSoup
URL = "http://tesaurus.kemdikbud.go.id/tematis/lema/"


def check_with_tesaurus(word):
      r = requests.get(url = URL+word)
      soup = BeautifulSoup(r.text, 'html.parser')
      cls_res_area = soup.find_all('div',{'class','note-notfound'})
      return len(cls_res_area)==0