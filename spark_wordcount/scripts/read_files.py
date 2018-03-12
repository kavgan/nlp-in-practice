import os
import re
from operator import add

from pyspark import SparkConf, SparkContext, SQLContext

datafile_json = "../sample-data/description.json"
datafile_csv = "../sample-data/description.csv"
datafile_psv = "../sample-data/description.psv"


def get_keyval(row):

    # get the text from the row entry
    text = row.text

    #remove unwanted chars
    text=re.sub("\\W"," ",text)

    # lower case text and split by space to get the words
    words = text.lower ().split (" ")

    # for each word, send back a count of 1
    return [[w, 1] for w in words]


def get_counts(df):
    # just for the heck of it, show 2 results without truncating the fields
    df.show (2, False)

    # for each text entry, get it into tokens and assign a count of 1
    # we need to use flat map because we are going from 1 entry to many
    mapped_rdd = df.rdd.flatMap (lambda row: get_keyval (row))

    # for each identical token (i.e. key) add the counts
    # this gets the counts of each word
    counts_rdd = mapped_rdd.reduceByKey (add)

    # get the final output into a list
    word_count = counts_rdd.collect ()

    # print the counts
    for e in word_count:
        print (e)


def process_json(abspath, sparkcontext):

    # Create an sql context so that we can query data files in sql like syntax
    sqlContext = SQLContext (sparkcontext)

    # read the json data file and select only the field labeled as "text"
    # this returns a spark data frame
    df = sqlContext.read.json (os.path.join (
        abspath, datafile_json)).select ("text")

    # use the data frame to get counts of the text field
    get_counts(df)


def process_csv(abspath, sparkcontext):

    # Create an sql context so that we can query data files in sql like syntax
    sqlContext = SQLContext (sparkcontext)

    # read the CSV data file and select only the field labeled as "text"
    # this returns a spark data frame
    df = sqlContext.read.load (os.path.join (abspath, datafile_csv),
                               format='com.databricks.spark.csv',
                               header='true',
                               inferSchema='true').select ("text")

    # use the data frame to get counts of the text field
    get_counts (df)


def process_psv(abspath, sparkcontext):
    """Process the pipe separated file"""

    # return an rdd of the tsv file
    rdd = sparkcontext.textFile (os.path.join (abspath, datafile_psv))

    # we only want the "text" field, so return only item[1] after
    # tokenizing by pipe symbol.
    text_field_rdd = rdd.map (lambda line: [re.split ('\|', line)[1]])

    # create a data frame from the above rdd
    # label the column as 'text'
    df = text_field_rdd.toDF (["text"])

    # use the data frame to get counts of the text field
    get_counts (df)


if __name__ == "__main__":

    # absolute path to this file
    abspath = os.path.abspath(os.path.dirname(__file__))

    # Create a spark configuration with 20 threads.
    # This code will run locally on master
    conf = (SparkConf ()
            . setMaster("local[20]")
            . setAppName("sample app for reading files")
            . set("spark.executor.memory", "2g"))

    sc = SparkContext(conf=conf)

    # process the json data file
    process_json (abspath, sc)

    # process the csv data file
    process_csv (abspath, sc)

    # process the pipe separated data file
    process_psv (abspath, sc)
