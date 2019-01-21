from elasticsearch import Elasticsearch
from datetime import timedelta, datetime
import numpy as np
import numpy.random
import pandas as pd


def dataCollection(start_time):
    #origin = datetime(2018, 7, 19, 20, 0, 0)
    #start_time = datetime(2018, 8, 20, 14, 0, 0)
    # start_time = origin
    #diff = datetime.utcnow() - start_time
    #total_hours = int(np.ceil(diff.total_seconds() / 3600))

    #print("******", "Extracting tweets for hour", start_time, "******", sep=" ")
    gte = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    lt = (start_time + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S")
    es = Elasticsearch("<Elasticsearch_URL>")
    doc = {
        'size': 10000,
        'query': {
            'range': {
                '@timestamp': {
                    "gte": gte,
                    "lt": lt
                }
            }
        }
    }

    # Data for positive Tweets
    op_first_page = es.search(index="<opiod crisis>", doc_type='doc', body=doc, scroll='1m')
    sid = op_first_page['_scroll_id']
    op_scroll_size_total = op_scroll_size = op_first_page['hits']['total']
    op_final_hits = op_first_page['hits']['hits']
    #print("op_scroll_size :", op_final_hits)
    while (op_scroll_size > 0):
        # print("Scrolling...")
        op_new_page = es.scroll(scroll_id=sid, scroll='1m')
        new_hits = op_new_page['hits']['hits']
        # Update the scroll ID
        sid = op_new_page['_scroll_id']
        # Get the number of results that we returned in the last scroll
        op_scroll_size = len(op_new_page['hits']['hits'])
        # print("scroll size: " + str(op_scroll_size))
        op_final_hits = op_final_hits + new_hits

    json_opioidData = op_final_hits
    opioidData = [d['_source'] for d in json_opioidData]
    opioidData = pd.DataFrame(opioidData)
    opioidData['type'] = 'gun violence'
    opioidData['rt_text'] = opioidData['rt_text'].replace(np.nan, '', regex=True)
    opioidData['text'] = opioidData[['rt_text', 'text']].apply(lambda x: ' '.join(x), axis=1)

    # Data for negative Tweets
    non_op_first_page = es.search(index="negative_sample", doc_type='doc', body=doc, scroll='1m')
    sid = non_op_first_page['_scroll_id']
    non_op_scroll_size_total = non_op_scroll_size = 2 * op_first_page['hits']['total']
    non_op_final_hits = non_op_first_page['hits']['hits']
    # print("NT_scroll_size :", non_op_scroll_size)
    while (non_op_scroll_size > 0):
        # print("Scrolling...")
        non_op_new_page = es.scroll(scroll_id=sid, scroll='1m')
        new_hits = non_op_new_page['hits']['hits']
        # Update the scroll ID
        sid = non_op_new_page['_scroll_id']
        # Get the number of results that we returned in the last scroll
        non_op_scroll_size = len(non_op_new_page['hits']['hits'])
        # print("scroll size: " + str(non_op_scroll_size))
        non_op_final_hits = non_op_final_hits + new_hits

    json_negativeTweets = non_op_final_hits
    negativeTweets = [d['_source'] for d in json_negativeTweets]
    negativeTweets = pd.DataFrame(negativeTweets)
    negativeTweets['type'] = 'negative tweets'

    negativeTweets = negativeTweets[negativeTweets.text.str.contains("RT") == False].reset_index(drop=True)  # Remove all retweets


    # Concat both tweets
    tweets = pd.concat([opioidData, negativeTweets])
    tweets = tweets.iloc[np.random.permutation(len(tweets))].reset_index(drop=True)
    tweets['text'] = tweets['text'].apply(str)

    data = tweets[['type', 'text']]
    original_data = tweets[['id','user_screen_name','state', 'type','text']]
    #TODO dropna generates warning
    data.dropna(inplace=True)
    data['type'].value_counts()
    return start_time, data, op_scroll_size_total, non_op_scroll_size_total,original_data
