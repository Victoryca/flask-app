# Flask app for summarizing news articles
This project scrapes news articles from [The Sun](https://www.thesun.co.uk/) and produces a summary of information from the articles, which includes the date published, date updated, title, author, geographical entities and summarized content. 

The motivation of this project was to experiment with text summarization using Hugging Face Transformers - [BART](https://huggingface.co/docs/transformers/main/en/model_doc/bart) and model deployment using Flask.


## Set-up and Installation

Setting up a virtual environment is preferred to avoid compatibility issues between dependencies.
```
python -m venv env
source env/bin/activate
```

1. Installing 🤗 Transformers using one of [several ways](https://huggingface.co/docs/transformers/installation). 
  
    Installation with pip can be done using:
  
    `pip install transformers`

    `pip install transformers[torch]`
  
2. Installing spaCy using one of [several ways](https://spacy.io/usage)
  
    Installation with pip can be done using:
   
    `pip install -U pip setuptools wheel`
  
    `pip install -U spacy`
  
    `python -m spacy download en_core_web_sm`

Other libraries:

3. Flask
4. Beautiful Soup
5. Pandas 
6. Numpy

## How it works
Provide article url(s) in the text box and click on submit. Multiple urls should be separated by a comma.
Wait for results to be generated and displayed.

## Note
Anyone who wishes to use the services from The Sun has to comply with their Terms and Conditions.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
