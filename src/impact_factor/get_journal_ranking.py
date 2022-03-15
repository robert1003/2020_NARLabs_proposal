import requests
import sys
import pandas as pd
from bs4 import BeautifulSoup

if len(sys.argv) <= 2:
    print('''
    Function: Crawl impact factor related information from ideas.repec.org
    Usage: python3 get_journal_ranking.py [url] [output_csv_name]
    Example: python3 get_journal_ranking.py https://ideas.repec.org/top/top.journals.hindex.html hindex.csv
    ''')
    exit(1)

url = sys.argv[1]
output_csv_name = sys.argv[2]

soup = BeautifulSoup(requests.get(url).text, features='lxml')
table = soup.find('table', {'class': 'toplist'})

data = []
for row in table.find_all('tr'):
    try:
        data.append(list(map(lambda x: x.text, row.find_all('td'))))
    except Exception as e:
        print(e)
        print(row)

df = pd.DataFrame(data[1:], columns=data[0])
df.to_csv(output_csv_name, index=False)
