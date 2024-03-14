## 2024/3/14 
#### pytorch实现各类神经网络，使用mnist进行验证
- cnn
- rnn
- lstm
#### 正在完善的模型...
- transfomer
- AE

#### 注意事项
新数据库添加完成后，到data的__init__种加入DataFactory里
新模型都请继承BaseModel类，实现类的loss_function和metrics_function，然后加入ModelFactory里
然后修改run.py里的args参数即可运行
