import pymongo
import pandas as pd
import scipy as sp
import numpy as np
def establish_conn(coll_name='women_reviews'):
    client = pymongo.MongoClient()
    db = client['review_data']
    coll = db[coll_name]
    return coll


def fetch_data():
    conn = establish_conn()
    df = conn.find({}, {"_id": 0})
    df = pd.DataFrame(list(df))
    return df

