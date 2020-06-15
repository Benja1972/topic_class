from gensim.matutils import jensen_shannon , kullback_leibler, jaccard, cossim
from gensim.matutils import corpus2dense
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from sklearn.metrics.pairwise  import cosine_similarity
from os.path import join
import warnings
from scipy.spatial.distance import squareform
import numpy as np
warnings.filterwarnings("ignore")
from gensim import corpora
from joblib import dump, load


class dictionary(object):
    """
    Dictionary class. Dictionary can be created from list of token of from text. In many cases compatible with Dictionary from *gensim*
        
        Parameters
        ----------
        voc: list of tokens, it should be unique
        
        Attributes
        ----------
        id2token: standart dictionary of id2token
        token2id: standart dictionary of token2id

    """

    def __init__(self,voc):
        if len(set(voc)) == len(voc):
            self.id2token = {i:v for i,v in enumerate(voc)}
            self.token2id = {v:i for i,v in enumerate(voc)}
        else:
            raise ValueError('Not a vocabulary: elements are not unique')
    
    @classmethod
    def from_text(cls, text):
        """
        Create dictionary from raw text 
        
            Parameters
            ----------
            text: text as string of words separated by spaces 
        """
        voc = list(set(text.strip().split()))
        return cls(voc)
    
    def __getitem__(self,idt):
        return self.id2token[idt]
    
    def __len__(self):
        return len(self.id2token)
    
    def keys(self):
        return self.id2token.keys()
    
    def tokens(self):
        return self.token2id.keys()

class topic(object):
    """
    Basic topic class. Topic can be created from list of token and weights
        
        Paremeters
        ----------
        vec: vector of floats, 1D numpy array 
        voc: tokens of topic, list of tokens should be unique
        
        Atributes
        ---------
        vec: vector of floats, 1D numpy array normilized
        dcit: dictionary class 
        bow: extended bag-of-words with, list of tuples (id,weight)  with non-zero values of weights
    """
    def __init__(self, vec, voc, norm=True):

        if len(vec) == len(voc):
            if norm:
                vec = vec/vec.sum()
            self.vec=np.array(vec)
            self.dict = dictionary(voc)
            self.bow = [(k,self.vec[k]) for k in self.dict.id2token.keys() if self.vec[k]>0]
        else:
            raise ValueError('Sizes not matched')
    
    def top_n(self,topn=20, norm=True):
        """
        Return separated topic with topn tokens of original topic, weights are re-normilized
        
            Parameters
            ----------
            topn: top N tokens to return
            
            Return
            ------
            tp_out: topic class instance with only top N tokens
            
        """
        ss = self.strip()
        topn = min(topn,len(ss))
        ind = np.argsort(ss.vec)[::-1]
        vec = ss.vec[ind[:topn]]
        if norm:
            vec = vec/vec.sum()
        vocn = [ss.dict.id2token[i] for i in ind[:topn]]
        tp_out = topic(vec,vocn)
        return tp_out
    
    def strip(self):
        """
        Represent stripped version of topic removing all zero-weighted tokens
        """
        svec = np.array([bw[1] for bw in self.bow])
        svoc = [self.dict.id2token[bw[0]] for bw in self.bow]
        tp_out = topic(svec,svoc)
        return tp_out
    
    def __len__(self):
        return len(self.vec)
    
    def __repr__(self):
        out=''
        for i in range(self.__len__()):
            out+= '('+str(self.dict[i])+', '+str(self.vec[i])+')\n'
        return out


class TopicModel(object):
    """
    Topic model is combined set of topics with shared vocabulary
    
        Parameters
        ----------
        topics_list: 
        
        Atributes
        ---------
        top_dist: numpy array of  weights fro all topics from list 
        dict: shared vocabulary as dictionary class
        
    """
    def __init__(self,topics_list,word2vec=None):

        self.topics_list=topics_list
        self.num_topics=len(topics_list)
        vo = [list(tp.dict.tokens()) for tp in topics_list]
        voc = merge_vocab(vo)
        tl = [transform_topic(tp,voc) for tp in topics_list]
        
        self.top_dist =  np.vstack([tp.vec for tp in tl])
        self.dict = dictionary(voc)
        if word2vec is not None:
            self.word2vec=word2vec # word2vec model used by soft cosine similarity and embedding+cosine similarity moodel
            self.term_ind=WordEmbeddingSimilarityIndex(self.word2vec.wv)
 
    
    def __getitem__(self,idt):
        return topic(self.top_dist[idt],self.dict.tokens())
    
    def __len__(self):
        return self.top_dist.shape[0]
    
    def add_topic(self,topic, inplace=False):
        """
        Add topic to list
        
        Parameters
        ----------
        topic: topic to add
        inplace: boolean, if True current topic model will be updated
        
        Returns
        -------
        tmp: new topic model with additional topic in list
        """
        tp_list = self.to_list()
        tp_list.append(topic)
        
        tmp = TopicModel(tp_list)
        if inplace:
            self.top_dist = tmp.top_dist
            self.dict = tmp.dict
        else:
            return tmp
        
        
    def top_n(self,topn=20, norm=True):
        """
        Take  top_n for all topics individually and re-assemble them in new topic model

        Parameters
        ----------
        topn: top N tokens to return for each topic in model
         
        """
        topn_list = [tp.top_n(topn,norm) for tp in self]
        return TopicModel(topn_list)
    
    def to_list(self):
        """
        Return all topic as list
        """
        return [tp for tp in self]

    
    def classify(self, sent, metric='jaccard'):
        """
        Calculate similarity of sentence corresponding to each topic in model
        
        
        Parameters
        ----------
        sent: sentence in topic format, see text2sent() for details
        metric: metric to calculated similarity, could be
            - 'jaccard'
            - 'cosine'
            - etc.
        
        Return
        ------
        list of floats similarities
        
        """
        if metric=='jaccard':
            return [self.sim_jaccard(tp,sent) for tp in self]
        elif metric=='counts':
            jcs = [self.sim_jaccard(tp,sent) for tp in self]
            cnt = [1 if jc>1e-20 else 0 for jc in jcs]
            return cnt
        elif metric=='cosine':
            return [self.sim_cossim(tp,sent) for tp in self]
        elif metric=='soft-cosine':
            try:
                return [self.sim_softcos(tp,sent,self.term_ind) for tp in self]
            except AttributeError:
                print("provide word2vec model... and try again!!")

        elif metric=="embedding+cosine":
            try:
                return [self.sim_cosembed(tp,sent,self.word2vec) for tp in self.topics_list]
            except AttributeError:
                print("provide word2vec model... and try again!!")

        else:
            raise ValueError('Metric not implemented')


    def get_topic_distances(self,metric='jaccard'):
        topics_distances=[]
        for i in range(self.num_topics):
            for j in range(i+1,self.num_topics):
                if metric=='jaccard':
                    topics_distances.append(self.sim_jaccard(self.__getitem__(i),self.__getitem__(j)))
                elif metric=='cosine':
                    topics_distances.append(self.sim_cossim(self.__getitem__(i),self.__getitem__(j)))
                elif metric=='soft-cosine':
                    try:
                        topics_distances.append(self.sim_softcos(self.__getitem__(i),self.__getitem__(j),self.term_ind))
                    except AttributeError:
                        print("provide word2vec model... and try again!!")

                elif metric=="embedding+cosine":
                    try:
                        topics_distances.append(self.sim_cosembed(self.topics_list[i],self.topics_list[j],self.word2vec))
                    except AttributeError:
                        print("provide word2vec model... and try again!!")
              
                else:
                    raise ValueError('Metric not implemented')
        
        return squareform(np.array(topics_distances))


    @staticmethod
    def sim_cosembed(top,sent,word2vec):
        sent_embed=[]
        for word in list(sent.dict.tokens()):
            if word in word2vec.wv.vocab:
                sent_embed.append(word)
        text_embedding=average_embedding(word2vec,sent_embed)
        topic_embedding=average_embedding(word2vec,list(top.dict.tokens()))
        similarity=vectors_similarity(text_embedding,topic_embedding)
        return similarity

    @staticmethod
    def sim_softcos(top,sent,term_ind):

        voc = merge_vocab([top.dict.token2id.keys(),sent.dict.token2id.keys()])
        top_= transform_topic(top,voc)
        sent_= transform_topic(sent,voc)
        dic = top_.dict
        
        sim_mx = SparseTermSimilarityMatrix(term_ind, dic)

        top_bw = top_.bow
        snt_bw = sent_.bow

        sim = sim_mx.inner_product(top_bw, snt_bw, normalized=True)
        return (sim+abs(sim))/2

    @staticmethod
    def sim_jaccard(top,sent):
        dst = jaccard(top.strip().dict.tokens(),sent.strip().dict.tokens())
        return 1-dst
    
    @staticmethod
    def sim_cossim(top,sent):
        voc = merge_vocab([top.dict.tokens(),sent.dict.tokens()])
        top_= transform_topic(top,voc)
        sent_= transform_topic(sent,voc)

        sim = cossim(top_.bow,sent_.bow)
        return (sim+abs(sim))/2

    def save(self, file):
        """
        Saves the current model to the specified file.
        Parameters
        ----------
        file: str
            File where model will be saved.
        """
        dump(self, file)

    @classmethod
    def load(cls, file):
        """
        Load a pre-defined model from the specified file.
        Parameters
        ----------
        file: str
            File where model will be loaded from.
        """
        return load(file)


def merge_vocab(voc_list):
    """
    Provided list of vocabularies function merges them in one list of unique tokens
    
    Parameters
    ----------
    voc_list: list of lists of vocabularies
    
    Return
    -------
    list of merged tokens
    """
    return list(set().union(*voc_list))
            

def transform_topic(top,vocn):
    t2ido = top.dict.token2id
    lo = t2ido.keys()
    
    t2idn = {v:i for i,v in enumerate(vocn)}
    td = top.vec  
                                            
    tdn = np.zeros((len(t2idn,)))
    for l in lo:
        tdn[t2idn[l]] = td[t2ido[l]]

    topic_out = topic(tdn,vocn)
    return topic_out

#embedding + cosine similarity 
def average_embedding(embedding_model, tokens,weights=None):
    '''
    embedding_model: the word embedding model
    tokens: a list of tokens , type: list / tokens=['tok1',tok2,...]
    weigths: the distribution of tokens in topics or in text , type list (weights=[0,3,4,0])
    
    '''
    words=[]
    for word in tokens:
        if word in embedding_model.wv.vocab:
            words.append(word)

    embedding=np.array(embedding_model[words])
    if weights is None: # for alpha10x henome
        return np.mean(embedding, axis=0)
    else:# for the generated topics from topic extraction model 

        weights=np.array(weights)
        return np.mean(np.array([a*b for a,b in zip(embedding,weights)]),axis=0)
def vectors_similarity(a,b):
    '''
    cosine similarity
    a,b: vector embedding
    return the cosine similarity between two vector
    
    '''
    a=a.reshape(1,-1)
    b=b.reshape(1,-1)
    sim=cosine_similarity(a,b)[0][0]
    return (sim+abs(sim))/2




def text2sent(text, voc=None):
    if not voc:
        voc = corpora.Dictionary([text])
    
    bw = voc.doc2bow(text)
    ntr = len(voc)
    vv = corpus2dense([bw],num_terms=ntr).T[0]
    vvn = vv/vv.sum()
    sent = topic(vvn,voc.token2id.keys())
    return sent


if __name__ == "__main__":
    
    # topic tokens
    tokens =  [['camera','vision','light','algorithm','reconstruction','stereo','geometry','imaging','color','field'],
                ['video','action','motion','human','temporal','sequence','frame','recognition','model','scene'],
                ['clustering','algorithm','subspace','transform','proposed','technique','color','used','hashing','distance']]
    


    # pretrained gensim word2vec
    # == Load Word2vec and extract term index
    #w2v_mdl = Word2Vec.load("./models/w2v-cv.model")
    
    # topic weights
    w = [abs(np.random.randn(len(tk))) for tk in tokens]

    topics = [topic(w[i],tk) for i,tk in enumerate(tokens)]
    
    #tm = TopicModel(topics,w2v_mdl)
    tm = TopicModel(topics)
    topa =  tm[0]
    topb =  tm[2]
    topc = topic(np.array([0.1,0.3,0.4,0.5,0.1,0.001]),['add','topic','to','class', 'in', 'time'])
    
    tm.add_topic(topc, inplace=True)


    print("Topic A: \n", topa.strip())
    print("Topic B: \n", topb.strip())
    

    sent = "light survey frame opinion computer  video system response time".split()
    sent = text2sent(sent)
    print('Sent classify Jaccard:', tm.classify(sent))
    print('Sent classify Counts:', tm.classify(sent, metric='counts'))
    print('Sent classify Cosine:', tm.classify(sent, metric='cosine'))
    print('sent classify soft-cosine',tm.classify(sent,metric='soft-cosine'))
    print('sent classify embedding+cosine',tm.classify(sent,metric='embedding+cosine'))

    print(tm.get_topic_distances())
    print('Saving model')
    tm.save('tmp.bin')
    
    print('Loading  model')
    tm_l = TopicModel.load('tmp.bin')




