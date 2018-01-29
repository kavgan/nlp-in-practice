import gzip
import gensim
import logging
import os

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def show_file_contents(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)


if __name__ == '__main__':

    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "../reviews_data.txt.gz")

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    logging.info("Done reading data file")

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        size=150,
        window=10,
        min_count=2,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)

    w1 = "dirty"
    model.wv.most_similar(positive=w1)

    # look up top 6 words similar to 'polite'
    w1 = ["polite"]
    model.wv.most_similar(positive=w1, topn=6)

    # look up top 6 words similar to 'france'
    w1 = ["france"]
    model.wv.most_similar(positive=w1, topn=6)

    # look up top 6 words similar to 'shocked'
    w1 = ["shocked"]
    model.wv.most_similar(positive=w1, topn=6)

    # get everything related to stuff on the bed
    w1 = ["bed", 'sheet', 'pillow']
    w2 = ['couch']
    model.wv.most_similar(positive=w1, negative=w2, topn=10)

    # similarity between two different words
    model.wv.similarity(w1="dirty", w2="smelly")

    # similarity between two identical words
    model.wv.similarity(w1="dirty", w2="dirty")

    # similarity between two unrelated words
    model.wv.similarity(w1="dirty", w2="clean")

    # Which one is the odd one out in this list?
    model.wv.doesnt_match(["cat", "dog", "france"])

    # Which one is the odd one out in this list?
    model.wv.doesnt_match(["bed", "pillow", "duvet", "shower"])
