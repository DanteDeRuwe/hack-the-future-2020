# To install the libraries, use requirement.txt file.
# pip install -r requirement.txt
import streamlit as st
import twint
import pandas as pd
import re
import numpy as np
import datetime
import time
import os

# To use the transformers library you need to install pytorch.
# Follow instruction via this link: https://pytorch.org/
# For windows without CUDA: pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, AutoConfig


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
    text = re.sub(r'\@', '', text)  # remove mentions
    text = re.sub(r'\#', '', text)  # remove hashtags
    return text


def process_text(text):
    """Apply remove content & lower text"""
    text = remove_content(text)
    text = re.sub('[^a-zA-ZÀ-ÿ]', ' ', text.lower())  # remove non-alphabets
    return text


def get_political_parties():
    return os.listdir("politicians")


####### Streamlit Side panel #########
# side panel for twint configuration
add_text = st.sidebar.title(
    'Code Commanders\nPolitical Analyzer'
)

# Create date selection for streamlit
add_today = datetime.date.today()
add_lastweek = add_today - datetime.timedelta(days=7)

start_date = st.sidebar.date_input('Start date', add_lastweek)
end_date = st.sidebar.date_input('End date', add_today)

# Warning if dates order is not correct.
if start_date > end_date:
    st.sidebar.error('End date must fall after start date.')

party_select = st.sidebar.selectbox(
    'Partij',
    tuple(get_political_parties())
)

######## End sidebar #######


######## Streamit main #######
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')


# st.write(start_date, end_date)


# if keyword(s) for search, start showing graph in dashboard
def get_username_from_handle(handle):
    return handle.replace("@", "").strip()

if st.sidebar.button("Analyze!") and party_select:

    start = time.time()


    print('Searching now...')
    # Config Twint
    c = twint.Config()
    c.Limit = 50
    c.lang = "nl"
    c.Since = start_date
    c.Until = end_date
    c.Hide_output = True
    c.Pandas = True
    c.Popular_tweets = True

    data = []
    politicians = [p for p in open("politicians/" + str(party_select), "r")]
    for politician in politicians:
        c.Username = get_username_from_handle(politician)
        print("getting " + c.Username)
        twint.run.Search(c)
        data.append(twint.storage.panda.Tweets_df)

    df = pd.concat(data)
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

        # # Example plot: hour vs length
        # df_HL = df[["hour", "length"]]
        # df_HL = df_HL.groupby(['hour']).mean()
        # st.bar_chart(data=df_HL, width=0, height=0, use_container_width=True)

        # Example plot: top 10 hashtags
        hashes = df['hashtags'].tolist()
        tags = []
        for x in hashes:
            if x:
                tags.extend(x)
        df_tags = pd.DataFrame(tags, columns=['hashtag'])
        st.bar_chart(data=df_tags['hashtag'].value_counts().head(
            10), width=0, height=0, use_container_width=True)

        st.write("Summarizing tweets from: " + ', '.join(politicians))

        s = remove_content('.'.join(df['tweet']))

        MODEL = "wietsedv/bert-base-dutch-cased"
        custom_config = AutoConfig.from_pretrained(MODEL)
        custom_config.output_hidden_states = True
        custom_tokenizer = AutoTokenizer.from_pretrained(MODEL)
        custom_model = AutoModel.from_pretrained(MODEL, config=custom_config)

        from summarizer import Summarizer

        model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
        output = model(s, min_length=15, max_length=40)
        st.write(''.join(output))

        print(time.time() - start)
