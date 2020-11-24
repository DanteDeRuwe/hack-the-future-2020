
# Hack The Future 2020 (AI Application Challenge)
<img height="100px" src="https://i.imgur.com/9jy03IT.png"/>&nbsp;&nbsp;&nbsp;<img height="100px" src="https://i.imgur.com/IxJEaZH.png"/>
<br>

Our proof-of-concept solution for the Hack The Future 2020 AI Application challenge. 
[**We won**](https://www.linkedin.com/posts/dantederuwe_hackathon-hackathon2020-hackthefuture-activity-6737088028495425537-XgcK) 🥳! In an intense competition, our web application was rated highest by the panel of experts.

Hack the Future is an experience-driven hackathon for last year students in IT, powered by De Cronos Groep. We only got 7 hours to come up with an entire solution.

## Challenge
[🌐 Website](https://hackthefuture.be/2020)<br>
[📃 Problem Statement](https://git.cronos.be/xplodata/hackthefuture2020/blob/master/HackTheFuture_XploData.pdf)

#### Short Summary:
> You are challenged to build an application that makes use of natural language processing. The problem that the NLP model should solve can be decided by all of you who choose this challenge. It is expected to deliver a working prototype application at the end of the day, which can be done via a web application. The group has some flexibility depending on the time they have. You’ll have to code in order to train and/or call a NLP model. Some understanding on how to build and/or code an application to put a machine learning model into production is key.
> In this project, there are different phases involved that you would encounter during a project with a customer in real life (preparation, execution, production) and you’ll learn how to face those. The idea is also to allow you to get some hands on machine learning, an exciting and booming area in today’s industry and that could be directly applied to a real situation. The challenge is also to get inspired by the potential that machine learning can have if you build a good case with it.

## Solution
### Concept
A web application that makes use of natural language processing to analyze political parties and their social media presence on Twitter. Our prototype included all major Flemish political parties.

### Technical
We used:
- [Python](https://python.org)
- [Pandas](https://pandas.pydata.org/)
- [Twint](https://github.com/twintproject/twint) (Twitter Intelligence Tool)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [BERTje](https://github.com/wietsedv/bertje) (a Dutch pre-trained BERT model)
- [Summarizer](https://github.com/dmmiller612/bert-extractive-summarizer) (bert-extractive-summarizer)
- [Streamlit](https://www.streamlit.io/) for the web UI

### Try it out 

#### Install
Make sure Python 3 or above is installed.
You also need to install all the necessary libraries define in the script. To install some libraries, use the pip command below:
```cmd
pip install pandas 
pip install streamlit 
pip install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
pip install bert-extractive-summarizer
```

To use the transformers library you need to install [PyTorch](https://pytorch.org/).

#### Run 
```bash
streamlit run main.py
```

## Contributors
- [Liam Spitaels](https://github.com/liamspitaels)
- [Dante De Ruwe](https://github.com/dantederuwe)

Team name: The Code Commanders

## Aknowledgements

Special thanks to Jean-Joseph Adjizian, Reinert Roux and Peter Boschmans and the following organizations as a whole:
- [hack The Future](https://hackthefuture.be)
- [De Cronos Groep](https://cronos-groep.be/)
- [XploData](https://xplodata.be/)

