import numpy as np
from topic_class import TopicModel, topic, dictionary, text2sent

# Topic tokens
tokens =  [['camera','vision','light','algorithm','reconstruction','stereo','geometry','imaging','color','field'],
            ['video','action','motion','human','temporal','sequence','frame','recognition','model','scene'],
            ['clustering','algorithm','subspace','transform','proposed','technique','color','used','hashing','distance']]


# Topic weights
w = [abs(np.random.randn(len(tk))) for tk in tokens]

topics = [topic(w[i],tk) for i,tk in enumerate(tokens)]

# Create model
tm = TopicModel(topics)

# Basic class topic is used to create single topic
top_new = topic(np.array([0.1,0.3,0.4,0.5,0.1,0.001]),['add','topic','to','class', 'in', 'time'])

# topic class is very sophisticated, it has 
#- its own dictionary, 
#- can be extended to new dictionary, 
#- can be striped with removing zero-weighted entries, 
#- it has property of bag-of-word, 
#- one can extract top_n words sorted by weights and re-weighted

print('Top 5 words of topic\n', top_new.top_n(5))


# TopicModel class and topic class contain dictionary class to keep vocabulary of class in enumerated order presenting id2token and token2id as atributes
# dictionary class also provide ability to created dictionary from text

dic = dictionary.from_text('dict from text is easy to create')

print('Tokens of dictionary', dic.id2token)


# Add new topic to existing model
tm.add_topic(top_new, inplace=True)

print('Topics from topic model')
print("Topic A: \n", tm[2].strip())
print("Topic B: \n", tm[0].strip())



# External text can be analysed based on several similarity metrics for all topics in TopicModel
sent = "light survey frame opinion computer  video system response time".split()

# Function text2sent() helps to translate text to weighted bag-of-words 
sent = text2sent(sent)

# Similarity of text to all topics of TopicModel are calculated 
print('Sentence classify Jaccard:', tm.classify(sent))
print('Sentence classify Cosine:', tm.classify(sent, metric='cosine'))

# Model can be saved for later use
print('Saving model')
tm.save('tmp.bin')

# Model can be loaded from disk
print('Loading  model')
tm_l = TopicModel.load('tmp.bin')
