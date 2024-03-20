

class BaseServer():

    def __init__(self, Model, aggregator):
        self.Model = Model
        self.aggregator = aggregator
        self.clients = []

    def evalute(self, datasets):

        pass

    def test(self, datasets):

        pass

    def connect(self, req_data):
        resp_data = {}
        return resp_data

    