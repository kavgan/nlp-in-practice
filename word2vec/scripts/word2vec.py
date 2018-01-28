import argparse
import gzip

import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




def read_input(input_file):

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open (input_file, 'rb') as f:
        for i, line in enumerate (f):

            if (i%5000==0):
                logging.info ("read {0} reviews".format (i))

            yield gensim.utils.simple_preprocess (line)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input tsv')
    parser.add_argument('output', help='model output')

    parser.add_argument('--randomize',
                        help='also add two randomized versions of document '
                             'bodies, helpful for regulating e.g. activity '
                             'streams',
                        const=True, default=False, action='store_const')
    parser.add_argument('--save',
                        help='Save model to disk at the end of each training '
                             'epoch',
                        const=True, default=False, action='store_const')
    parser.add_argument('--epochs',
                        help='Number of training epochs',
                        default=10, type=int)
    parser.add_argument('--min_count',
                        help='Ignore all words with total frequency lower '
                             'than this',
                        default=5, type=int)
    parser.add_argument('--window',
                        help='The maximum distance between the predicted word '
                             'and context words used for prediction within a '
                             'document',
                        default=5, type=int)

    args = parser.parse_args()

    logging.info('Loading Input')
    documents = list (read_input (args.input))
    model = gensim.models.Word2Vec (documents, size=150, window=2, min_count=2, workers=10)
    model.train(documents,total_examples=len(documents),epochs=10)

    w1 = "noise"
    print (w1, model.wv.most_similar (positive=[w1]))

    w1 = "shower"
    print (w1, model.wv.most_similar (positive=[w1]))

    w1 = ["shower",'bed']
    print (w1, model.wv.most_similar (positive=w1))

    w1 = ["cheap"]
    print (w1, model.wv.most_similar (positive=w1))

    w1 = ["affordable"]
    print (w1, model.wv.most_similar (positive=w1))

    logging.info('Model Save')
    model.save(args.output)
