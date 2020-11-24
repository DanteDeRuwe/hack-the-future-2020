# To install the libraries, use requirement.txt file.
# pip install -r requirement.txt
import streamlit as st
import twint
import pandas as pd
import re
import numpy as np
import datetime
import time

# To use the transformers library you need to install pytorch.
# Follow instruction via this link: https://pytorch.org/
# For windows without CUDA: pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
# from transformers import T5ForConditionalGeneration, T5Tokenizer


# Command example for twint outside python script.
# twint -s 'Belgium covid lang:en' --limit 100 --output belgium.csv --csv


def get_hashtags(text):
    """Function to extract hashtags from tweets"""
    hashtags = re.findall(r'\#\w+', text.lower())
    return hashtags


def get_mentions(text):
    """Function to extract mentions from tweets"""
    mentions = re.findall(r'\@\w+', text.lower())
    return mentions


def remove_content(text):
    """Clean up text by removing urls, mentions & hashtags"""
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub(r'\S+\.com\S+', '', text)  # remove urls
    text = re.sub(r'\@\w+', '', text)  # remove mentions
    text = re.sub(r'\#\w+', '', text)  # remove hashtags
    return text


def process_text(text):
    """Apply remove content & lower text"""
    text = remove_content(text)
    text = re.sub('[^a-zA-ZÀ-ÿ]', ' ', text.lower())  # remove non-alphabets
    return text


####### Streamlit Side panel #########
# side panel for twint configuration
add_text = st.sidebar.title(
    'Config the news'
)

# Create date selection for streamlit
add_today = datetime.date.today()
add_yesterday = add_today - datetime.timedelta(days=1)

start_date = st.sidebar.date_input('Start date', add_yesterday)
end_date = st.sidebar.date_input('End date', add_today)

# Warning if dates order is not correct.
if start_date > end_date:
    st.sidebar.error('End date must fall after start date.')

# add_text_search = st.sidebar.write('Enter your search keyword:')
add_search = st.sidebar.text_input('Enter you search', '')
######## End sidebar #######


######## Streamit main #######
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')
# st.write(start_date, end_date)

# Config Twint
c = twint.Config()
c.Limit = 100
c.lang = "en"
c.Since = start_date
c.Until = end_date
c.Search = add_search
c.Hide_output = True
c.Pandas = True
c.Popular_tweets = True
#

# if keyword(s) for search, start showing graph in dashboard
if add_search:
    print('Searching now...')
    twint.run.Search(c)
    df = twint.storage.panda.Tweets_df
    time.sleep(10)
    print("Done")

    if len(df) > 1:
        df['date'] = pd.to_datetime(df['date'])

        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['length'] = df['tweet'].apply(lambda x: len(x.split()))
        df['hashtags'] = df['tweet'].apply(lambda x: get_hashtags(x))
        df['mentions'] = df['tweet'].apply(lambda x: get_mentions(x))
        df['num_hashtags'] = df['tweet'].apply(lambda x: len(get_hashtags(x)))
        df['num_mentions'] = df['tweet'].apply(lambda x: len(get_mentions(x)))
        df['hour'] = df['date'].dt.hour

        # Example plot: hour vs length
        df_HL = df[["hour", "length"]]
        df_HL = df_HL.groupby(['hour']).mean()
        st.bar_chart(data=df_HL, width=0, height=0, use_container_width=True)

        # Example plot: top 10 hashtags
        hashes = df['hashtags'].tolist()
        tags = []
        for x in hashes:
            if x:
                tags.extend(x)
        df_tags = pd.DataFrame(tags, columns=['hashtag'])
        st.bar_chart(data=df_tags['hashtag'].value_counts().head(
            10), width=0, height=0, use_container_width=True)

        df = df[['tweet']]
        df['cleaned_tweets'] = df['tweet'].apply(lambda x: process_text(x))
        df['cleaned_tweets'] = df['cleaned_tweets'].str.rstrip().str.lstrip()
        s = '.'.join(df.loc[:, 'cleaned_tweets'])

        # # Example NLP with transformers (transformer + pytorch)
        # # initialize the model architecture and weights
        # model = T5ForConditionalGeneration.from_pretrained("t5-small")

        # # initialize the model tokenizer
        # tokenizer = T5Tokenizer.from_pretrained("t5-small")

        # inputs = tokenizer.encode(
        #     "summarize: " + s, return_tensors="pt", max_length=512, truncation=True)

        # # generate the summarization output
        # outputs = model.generate(
        #     inputs,
        #     max_length=150,
        #     min_length=40,
        #     length_penalty=2.0,
        #     num_beams=4,
        #     early_stopping=True)
        # st.write(tokenizer.decode(outputs[0]))


#### Extra streamlit for info ####
# Add drop down menu
# add_selectbox = st.sidebar.selectbox(
    # 'How would you like to be contacted?',
    # ('Twitter', 'Mail', 'Mobile phone')
# )

# Add a slider to the sidebar:
# add_slider = st.sidebar.slider(
    # 'Select a range of values',
    # 0.0, 100.0, (25.0, 75.0)
# )