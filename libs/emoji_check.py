import requests
from unidecode import unidecode

from bs4 import BeautifulSoup
URL = "http://kt.ijs.si/data/Emoji_sentiment_ranking/index.html"
r = requests.get(url = URL)
soup = BeautifulSoup(r.text, 'html.parser')  
def _lookup(word):
      td = soup.find("td", text='0x'+word)
      try:
        negative = td.find_next_sibling("td").find_next_sibling("td").find_next_sibling("td").text
        neutral = td.find_next_sibling("td").find_next_sibling("td").find_next_sibling("td").find_next_sibling("td").text
        
        positive = td.find_next_sibling("td").find_next_sibling("td").find_next_sibling("td").find_next_sibling("td").find_next_sibling("td").text
        sent_score = td.find_next_sibling("td").find_next_sibling("td").find_next_sibling("td").find_next_sibling("td").find_next_sibling("td").find_next_sibling("td").text
        
        return float(negative), float(neutral), float(positive), float(sent_score)#((float(sent_score) - (-1)) / (1-(-1)),4)
      except:
          return None, None, None, None
    #   return len(cls_res_area)==0
def check_value(word):
    k = word.encode('unicode-escape').decode('ASCII')
    # print(k[5:])
    return _lookup(k[5:])

# n,nu,p,sc=check_value("😂")
# print(n,nu,p,sc)