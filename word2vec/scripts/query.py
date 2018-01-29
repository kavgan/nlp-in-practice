import gensim
import gensim.models.word2vec


model2 = gensim.models.Word2Vec.load("../models/model_win_2")
model = gensim.models.Word2Vec.load("../models/model")


word_list = [
    'gross',
    'dirty',
    'location',
    'breakfast',
    'smelly',
    'affordable',
    'hotel staff',
    'manager rude',
    'complimentary',
    'family',
    'awe',
    'shocked',
    'king size']


for w in word_list:
    sim = model.most_similar(positive=w.split(" "))
    sim2 = model2.most_similar(positive=w.split(" "))

    print("========", w)
    print(sim)
    print(sim2)
