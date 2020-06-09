# TopicModel class

Topic modeling using different approaches produce topics in form of topic-words distribution. This package helps to manipulate with topics and organize them in one TopicModel class with shared dictionary. 


## Getting Started

### Install TODO

Python code for the TopicModel can be installed via pip:

```
pip install {link to git repo}
```


### Running the TopicModel example

Given a set independent topics with topic-words distribution, with TopicModel it is easy to combine them in one pseudo-model. 

```python
import numpy as np
from topic_class import TopicModel

# topic tokens
tokens =  [['camera','vision','light','algorithm','reconstruction','stereo','geometry','imaging','color','field'],
            ['video','action','motion','human','temporal','sequence','frame','recognition','model','scene'],
            ['clustering','algorithm','subspace','transform','proposed','technique','color','used','hashing','distance']]


# topic weights
w = [abs(np.random.randn(len(tk))) for tk in tokens]

topics = [topic(w[i],tk) for i,tk in enumerate(tokens)]

# create model
tm = TopicModel(topics)

# add new topic
topc = topic(np.array([0.1,0.3,0.4,0.5,0.1,0.001]),['add','topic','to','class', 'in', 'time'])
tm.add_topic(topc, inplace=True)


print("Topic A: \n", tm[2].strip())
print("Topic B: \n", tm[0].strip())


sent = "light survey frame opinion computer  video system response time".split()
sent = text2sent(sent)
print('Sent classify Jaccard:', tm.classify(sent))
print('Sent classify Cosine:', tm.classify(sent, metric='cosine'))

print('Saving model')
tm.save('tmp.bin')

print('Loading  model')
tm_l = TopicModel.load('tmp.bin')
```
