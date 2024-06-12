from pymongo import MongoClient
MONGO_URI = 'mongodb://localhost:27017'

def dbConnection():
    try:
        client = MongoClient(MONGO_URI)
        db = client["db_translation_log"]
    except ConnectionError   as e:
        print(e)
    return db