# Imports

from bs4 import BeautifulSoup
import requests

import random
import time

import numpy as np
import pandas as pd

import spacy
nlp = spacy.load("en_core_web_sm")

import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

from flask import Flask, request, render_template

# Create a Flask object named app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



# Define function for retrieving html content of a given webpage
def get_soup(URL):
    
    headers = {
    'authority': 'https://www.thesun.co.uk/',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'referer': 'https://www.thesun.co.uk/',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100", "Google Chrome";v="100"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'}
    
    r = requests.get(URL, headers=headers)
    
    print(r.url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")

    return soup

# Define function for scraping multiple links
def get_article_soup(link_list):

    soup_list = []

    for link in link_list:
   
        soup = get_soup(link)
        
        soup_list.append(soup)
        
        num = random.randint(1,5)
        time.sleep(20*num)   # Delay for random seconds
    
    return soup_list

# Function for obtaining article titles
def get_titles(soup_list):

    titles = []

    for soup in soup_list:

        title = soup.find("h1", "article__headline").getText()
        title = title.strip()
        titles.append(title)
        
    return titles

# Function for obtaining text content of articles from their html content
def get_contents(soup_list):
    
    all_contents = []

    for soup in soup_list:

        div = soup.find("div", "article__content")
        p_tags = div.find_all("p", class_=None)

        content = []
        
        # Retrieve content within each p tag
        for tag in p_tags:
            text = tag.getText()
            
            if not text.startswith('Email us') and \
            not text.startswith('Click here') and not text.startswith('We pay for your stories and videos!'):
                
                content.append(text)

        content = " ".join(content).strip()

        all_contents.append(content)

    return all_contents

# Function for obtaining date & time of publish and update
def get_datetime(soup_list):
    
    date_time = []

    for soup in soup_list:

        published = soup.find("li", "article__published").getText()
        
        try:
            updated = soup.find("li", "article__updated").getText()
        except:
            updated = ''
        else:
            updated = updated.split(':', 1)[1].strip()

        
        date_time.append([published, updated])
        

    return date_time

# Function for extracting the author 
def get_author(soup_list):
    
    authors = []
    for soup in soup_list:
        author = soup.find("li", "article__author-list t-p-background-color__before").getText()
        authors.append(author)
        
    return authors

# Function for extracting geographical locations/entities from text content
def country_by_content(content):

    doc = nlp(content)

    region_dic = {}
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            if ent.text in region_dic.keys():
                region_dic[ent.text] += 1
            else:
                region_dic[ent.text] = 1
        else:
            continue
    
    if len(region_dic)>0:
        # Sort geographical locations by number of occurences in descending order 
        sorted_region = sorted(region_dic, key = region_dic.get, reverse=True)
        sorted_region = ", ".join(sorted_region)
    else:
        return ""
    
    return sorted_region

# Function for summarizing text contents
def summarize_content_bart(content_list):
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    summary_list = []
    
    for content in content_list:
        
        # Encode the contents in an article
        inputs = tokenizer.batch_encode_plus([content], 
                                            max_length=1024,
                                            truncation = True,
                                            return_tensors="pt")['input_ids']

        # Generate summary
        outputs = model.generate(inputs, 
                                 num_beams=4, 
                                 max_length=150, 
                                 min_length=50)
        
        # Decode the generated summary
        summary = tokenizer.decode(outputs.squeeze(), 
                                   skip_special_tokens=True, 
                                   clean_up_tokenization_spaces=True)

        # Append each summary in the list
        summary_list.append(summary)
    
    return summary_list

# Consolidate all results into a DataFrame
def results(titles, authors, date_time, regions, contents_summ, link_list):
    data = list(map(list, zip(titles, authors, regions, contents_summ, link_list)))
    data = np.hstack((np.array(date_time), data))

    df = pd.DataFrame(data = data, columns = ['Date Published', 'Date Updated', 'Title',
                                          'Author', 'Regions', 'Summarized Contents', 'Link'])
    
    return df



# Define endpoint

@app.route('/summarize', methods=['POST'])
def summarize():
    
    if request.method == 'POST':
        
        # String containing article links
        inputs = request.form['links']
        
        # inputs = input('Please provide article links.')
    
        link_list = [link.strip() for link in inputs.split(',')]
        link_list = [link for link in link_list if link !='']

        soup_list = get_article_soup(link_list)
        titles = get_titles(soup_list)
        contents = get_contents(soup_list)
        date_time = get_datetime(soup_list)
        authors = get_author(soup_list)

        regions = []
        for content in contents:
            region = country_by_content(content)
            regions.append(region)
    
    
        contents_summ = summarize_content_bart(contents)
        result = results(titles, authors, date_time, regions, contents_summ, link_list)

    return render_template('summarize.html', result = result)
    


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
