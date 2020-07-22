# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:04:39 2020

@author: LENOVO
"""
"""
Implementation of a web scraping algorithm for collecting the dataset needed to train several machine learning models
"""

### Imports

import numpy as np
import random
import pandas as pd
import tensorflow as tf
import bs4
from urllib.request import urlopen

### Collecting data from the page trip advisor listing several restaurants reviews to train our scraping skills

import requests
raw_html2 = urlopen('https://www.tripadvisor.fr/Restaurant_Review-g1136257-d8599271-Reviews-Le_Scoop-Yffiniac_Cotes_d_Armor_Brittany.html')
page_html2 = raw_html2.read()
raw_html2.close()
soup = bs4.BeautifulSoup(page_html2,'html.parser')
soup.find('span',{'class': 'noQuotes'})
soup.div['class']

container = soup.find_all('span', {'class' : "noQuotes"})
reviews = []
filename = 'reviews.csv'
f = open(filename,'w', encoding="utf-8")
headers = 'reviews::target'
f.write(headers + '\n')

for element in container:
    review = element.text
    reviews.append(review)
    f.write(review + '::' + 'None' + '\n')  ### We use :: as a seprator because a coma can be present in several reviews, breaking the scraping
    
### To get more reviews, we scrap on several pages    
    
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
    
### There is a pattern in the way the url are written so we can use it to create a loop over all the pages   
for i in range(0,148):
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
        
       
f.close()

### Now we have to create our labels. To do this, we just have to label the bad review with a zero (that's quite fast because there are not many bad reviews).
### And then we'll do a loop over our dataframe to fill the rest of the reviews with one. It would mean that thay are positive reviews.
