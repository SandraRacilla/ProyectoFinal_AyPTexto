import urllib3
from bs4 import BeautifulSoup
from googlesearch import search

http = urllib3.PoolManager()

response = search("'nazis', 'mob', 'riots', 'don', 'law', 'just', 'people', 'attack', 'police', 'trump'", num_results = 1)
for result in response:
	url = result
	# Fetch the html file
	response = http.request('GET', url)
	soup = BeautifulSoup(response.data,'html.parser')
	print("\nURL: " + result)
	print("Title: " + soup.title.string, end='\n\n')
    