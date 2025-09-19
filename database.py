from pymongo import MongoClient
MONGO_URI = 'url'

def dbConnection():
    try:
        client = MongoClient(MONGO_URI)
        db = client["signapp"]
    except ConnectionError   as e:
        print(e)
    return db
