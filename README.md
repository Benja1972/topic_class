# TopicModel class

Topic modeling using different approaches produce topics in form of topic-words distribution. This package helps to manipulate with topics and organize them in one TopicModel class with shared dictionary.

## Getting Started

### Install TODO

Python code for the TopicModel can be installed via pip:

```sh
pip install git+https://github.com/Benja1972/topic_class.git
```

### Running the TopicModel example

Given a set independent topics with topic-words distribution, with TopicModel it is easy to combine them in one pseudo-model.

```python
import numpy as np
from topic_class import TopicModel, topic, dictionary, text2sent

# Topic tokens
tokens =  [['camera','vision','light','algorithm','reconstruction','stereo','geometry','imaging','color','field'],
            ['video','action','motion','human','temporal','sequence','frame','recognition','model','scene'],
            ['clustering','algorithm','subspace','transform','proposed','technique','color','used','hashing','distance']]


# Topic weights
w = [abs(np.random.randn(len(tk))) for tk in tokens]

topics = [topic(w[i],tk) for i,tk in enumerate(tokens)]
```

Basic class *topic* is used to create single topic
*topic* class is very sophisticated, it has:

- its own *dictionary*,
- can be extended to new dictionary,
- can be striped with removing zero-weighted entries,
- it has property of bag-of-word,
- one can extract *top_n* words sorted by weights and re-weighted

Now we can create a *TopicModel*

```python
# Create model
tm = TopicModel(topics)
```

We create also single topic

```python
# Basic class topic is used to create single topic
top_new = topic(np.array([0.1,0.3,0.4,0.5,0.1,0.001]),['add','topic','to','class', 'in', 'time'])

print('Top 5 words of topic\n', top_new.top_n(5))

# Add new topic to existing model
tm.add_topic(top_new, inplace=True)

```

 *TopicModel* class and *topic* class contain *dictionary* class to keep vocabulary of class in enumerated order presenting id2token and token2id as attributes
 dictionary class also provide ability to created dictionary from text

```python
dic = dictionary.from_text('dict from text is easy to create')

print('Tokens of dictionary', dic.id2token)

print('Topics from topic model')
print("Topic A: \n", tm[2].strip())
print("Topic B: \n", tm[0].strip())

```

External text can be analyzed based on several similarity metrics for all topics in TopicModel

```python
sent = "light survey frame opinion computer  video system response time".split()

# Function text2sent() helps to translate text to weighted bag-of-words 
sent = text2sent(sent)

```

Similarity of text to all topics of TopicModel are calculated

```python
print('Sentence classify Jaccard:', tm.classify(sent))
print('Sentence classify Cosine:', tm.classify(sent, metric='cosine'))

```

Model can be saved for later use

```python
print('Saving model')
tm.save('tmp.bin')

```

Model can be loaded from disk

```python
print('Loading  model')
tm_l = TopicModel.load('tmp.bin').
```
