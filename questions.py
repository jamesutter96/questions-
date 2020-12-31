import nltk
import sys
import os 
import pandas as pd 
import string
import math as m 



FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    # create the intial dictionary 
    intial_topics = {}

    # Defines the path to directory
    path = os.path.join(".", f"{directory}")

    # iterater over the files in the specified directory 
    for file in os.listdir(path):
        # get the file path
        file_path = os.path.join(path, file)

        # read the file into a string 
        with open(file_path, 'r', encoding = 'utf8') as f:
            string = f.read()

        # saving the text from each of the files and excluding the ".txt" at the end 
        intial_topics[file[:-4]] = string

    # return the dictionary 
    return intial_topics

    #raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # finding the tokens and seves them in a lowercase list 
    tokens = [word.lower() for word in nltk.word_tokenize(document)]

    # intialize a filter list of words that are not stop words 
    filter_list = []

    # initialize stop words and punctuations
    stop_words  = nltk.corpus.stopwords.words("english")
    punctuation = [punct for punct in string.punctuation] 

    # go over all the words anf filter out the stop words and punctuation 
    for word in tokens:
        if word in stop_words:
            continue 

        elif word in punctuation:
            continue 

        else:
            filter_list.append(word)


    return filter_list
    #raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # number of dictionaries or files 
    number_of_dictionary = len(documents)

    # intialize a list documents that a specific word in found in 
    presence_dictionary  = {}

    # initialize a dictionary that idfs' are in 
    idfs_dictionary      = {}

    # go through each document and find the unique words in them 
    for d in documents:
        for word in set(documents[d]):
            if word in presence_dictionary.keys():
                presence_dictionary[word] += 1

            else:
                presence_dictionary[word] = 1

    for word in presence_dictionary:
        # adding 1 smooths the values for each of the idfs 
        idf = 1 + (m.log(number_of_dictionary/(presence_dictionary[word])))
        idfs_dictionary[word] = idf


    return idfs_dictionary

    #raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    # initialize the top files 
    top_files = {}

    # go through all files and words in the quarery to rank all the files and create file-word pairs 
    for f in files:
        
        top_files[f] = 0
        tokens_in_file = len(files[f])

        for word in query:
            if word in files[f]:
                freq = files[f].count(word) + 1 # again for smoothing 

            else:
                freq = 1

            # normalize the frequency because the length of texts are different 
            normalize_freq = freq / tokens_in_file

            if word in idfs.keys():
                idf = idfs[word]

            else:
                idf = 1

            temp = idf * normalize_freq
            

            top_files[f] += temp

    # Create a list with sorted files (highest rank to lowest rank)
    ranked_list    = sorted(top_files, key = top_files.get, reverse = True)

    # creates a list with the top n files
    highst_n_files = ranked_list[:n]

    return highst_n_files

    #raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    # initialize the list of the highest ranked sentences 
    top_sentences = {}

    # itorate over the files and words again and calculate the rank of each file-word pair
    for sentence in sentences:
        top_sentences[sentence] = {}
        top_sentences[sentence]['idf'] = 0
        top_sentences[sentence]['word_count'] = 0
        sen_len = len(sentences[sentence])

        for word in query:
            if word in sentences[sentence]:
                top_sentences[sentence]['idf'] += idfs[word]
                top_sentences[sentence]['word_count'] += 1

        top_sentences[sentence]['QTD'] = float(top_sentences[sentence]['word_count'] / sen_len)


    ranked_list = sorted(top_sentences.keys(), key = lambda sentence: (top_sentences[sentence]['idf'], top_sentences[sentence]['QTD']), reverse = True)

    # created a list of the top n sentences 
    highest_n_sentences = ranked_list[:n]

    return highest_n_sentences
    
    #raise NotImplementedError


if __name__ == "__main__":
    main()
