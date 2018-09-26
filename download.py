from bs4 import BeautifulSoup
import urllib
import re


url = 'http://zju-capg.org/myo/data/'
store_url = 'C:\\Users\\robin\\Documents\\srt\\data\\download_file'
res = urllib.request.urlopen(url)
html = res.read().decode('utf-8')
#print(html)
soup = BeautifulSoup(html,'html5lib')
for tag  in soup.find_all('a'):
    if(re.match('dbc',tag['href'])):
        print(tag)
        name = tag['href']
        download_url = url + name
        urllib.request.urlretrieve(download_url, store_url+'\\'+name)
        print(name+' download done')