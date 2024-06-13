from pymongo import MongoClient
MONGO_URI = 'mongodb+srv://sebastian:jijijija@cluster0.wbpwbgo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

def dbConnection():
    try:
        client = MongoClient(MONGO_URI)
        db = client["signapp"]
    except ConnectionError   as e:
        print(e)
    return db