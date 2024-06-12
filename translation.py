class translation:
    def __init__(self, text, date):
        self.text = text
        self.date = date

    def toDbCollection(self):
        return {
            "text": self.text,
            "date": self.date
        }