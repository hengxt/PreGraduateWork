from datetime import datetime

class BaseClient():

    def __init__(self, model:BaseModel, server:BaseServer):
        self.clientId = None
        self.model = model
        self.remote = server
        self.createAt = datetime.timestamp(datetime.now())
        self.updateAt = datetime.timestamp(datetime.now())
        self._check_remote()

    def _check_remote(self):
        req_data = {
            'model': self.model,
        }
        try:
            resp = self.remote.connect(req_data)
            self.clientId = resp.data['clientId']
            self.updateAt = datetime.fromtimestamp(resp.data['updateAt'])
        except:
            raise Exception(resp.msg)

    def train_local(self, datasets):
        self.model.train(datasets)
        return self.model.metrics


    def predict_local(self, inputs):
        result = self.model.predict(inputs)
        return result

    def push_remote(self):
        req_data = {
            'model': model.dict(),
            'clientId': self.clientId
        }
        
        pass

    def pull_remote(self):

        pass
    