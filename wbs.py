# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:04:39 2020

@author: LENOVO
"""
"""
Implementation of a sentiment analysis algorithm from collecting the data
"""

### Imports

import numpy as np
import random
import pandas as pd
import tensorflow as tf
import bs4
from urllib.request import urlopen

### Collecting data

raw_html = urlopen("https://www.tripadvisor.fr/Restaurants-g187147-Paris_Ile_de_France.html")
page_html = raw_html.read()
raw_html.close()
soup1 = bs4.BeautifulSoup(page_html,'lxml')
soup1

names_restaurants = []
names_container = soup1.find_all('div', {'class' : '_1llCuDZj'})
names_container
L = names_container[2].span.div.div.span.a['href'].split('-')
nr = ''
d = 0
f = 0
for j in range(0,len(L)):
    if L[j] == 'Reviews':
        d = j
        print(d)
    if L[j] == 'Paris_Ile_de_France.html':
        f = j
nr = L[d+1:f]

filename = 'restaurant.csv'
file = open(filename,'w')
headers = 'restaurants, review'
file.write(headers)

for i in range(1, len(names_container)):
    a_tag = names_container[i].span.div.div.span.a['href']
    #print(i)
    #print(a_tag)
    L = a_tag.split('-')
    nr = ''
    d = 0
    f = 0
    for j in range(0,len(L)):
        if L[j] == 'Reviews':
            d = j
        if L[j] == 'Paris_Ile_de_France.html':
            f = j
    nr = L[d+1:f]
    file.write(nr[0] + ','+'None'+'\n')
    print(nr[0])
    names_restaurants.append(nr)

file.close()
    
df = pd.read_csv('restaurant.csv')
    
### New Scraping

import requests
raw_html2 = urlopen('https://www.tripadvisor.fr/Restaurant_Review-g1136257-d8599271-Reviews-Le_Scoop-Yffiniac_Cotes_d_Armor_Brittany.html')
page_html2 = raw_html2.read()
raw_html2.close()
soup = bs4.BeautifulSoup(page_html2,'html.parser')
soup.find('span',{'class': 'noQuotes'})
soup.div['class']
soup

container = soup.find_all('span', {'class' : "noQuotes"})
container[0].text
reviews = []
filename = 'reviews.csv'
f = open(filename,'w', encoding="utf-8")
headers = 'reviews::target'
f.write(headers + '\n')

for element in container:
    review = element.text
    reviews.append(review)
    f.write(review + '::' + 'None' + '\n')
    
    
raw_html3 = urlopen('https://www.tripadvisor.fr/Restaurant_Review-g1136257-d8599271-Reviews-or10-Le_Scoop-Yffiniac_Cotes_d_Armor_Brittany.html') 
page_html3 = raw_html3.read()
raw_html3.close()
soup3 = bs4.BeautifulSoup(page_html3,'html.parser')
container = soup3.find_all('span',{'class' : 'noQuotes'})

for element in container:
    review = element.text
    reviews.append(review)
    f.write(review +'::'+'None'+'\n')


raw_html3 = urlopen('https://www.tripadvisor.fr/Restaurant_Review-g1136257-d8599271-Reviews-or20-Le_Scoop-Yffiniac_Cotes_d_Armor_Brittany.html') 
page_html3 = raw_html3.read()
raw_html3.close()
soup3 = bs4.BeautifulSoup(page_html3,'html.parser')
container = soup3.find_all('span',{'class' : 'noQuotes'})

for element in container:
    review = element.text
    reviews.append(review)
    f.write(review +'::'+'None'+'\n')
    
raw_html3 = urlopen('https://www.tripadvisor.fr/Restaurant_Review-g1136257-d8599271-Reviews-or30-Le_Scoop-Yffiniac_Cotes_d_Armor_Brittany.html') 
page_html3 = raw_html3.read()
raw_html3.close()
soup3 = bs4.BeautifulSoup(page_html3,'html.parser')
container = soup3.find_all('span',{'class' : 'noQuotes'})

for element in container:
    review = element.text
    reviews.append(review)
    f.write(review +'::'+'None'+'\n')
    
for i in range(1,148):
    if i == 0:
        raw_html = urlopen('https://www.tripadvisor.fr/Restaurant_Review-g187089-d1881625-Reviews-La_Vieille_Auberge-Saint_Jean_de_Luz_Basque_Country_Pyrenees_Atlantiques_Nouvelle.html')
    else:
        raw_html = urlopen('https://www.tripadvisor.fr/Restaurant_Review-g187089-d1881625-Reviews-or'+str(10*i)+'-La_Vieille_Auberge-Saint_Jean_de_Luz_Basque_Country_Pyrenees_Atlantiques_No.html')
    page_html = raw_html.read()
    raw_html.close()
    soup = bs4.BeautifulSoup(page_html,'html.parser')  
    container = soup.find_all('span',{'class': 'noQuotes'}) 
    for element in container:
        review = element.text
        print(review)
        reviews.append(review)
        f.write(review + '::' + 'None' + '\n')
        
        
### Label each bad review with 0    



f.close()

