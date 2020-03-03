import os
import codecs
import re
from cleaning_code_expanded import cleaning_code
from nltk.tokenize import word_tokenize

#This function will take in an article name, and output the text of the article
#Only keeps sentences with >=3 words, words with >= 3 chars, english words, and non-stop words
def import_article(fname1,english_words,stop_words):
    """ import_article function
    @param frame1 (string): name of the article
    @param english_words (list of strings): list of english words to keep
    @param stop_words (list of string): list of stop words to filter out
    @returns article2 (list of strings): a tokenized list of words (string) 
    """
    os.chdir('/Users/arjun/Documents/cs224n/deepjump/WSJ_txt')

    #print(fname1)

    with codecs.open(fname1, 'r', encoding='utf8', errors='replace') as myfile:
        ftext = myfile.read()

    article=ftext

    #Check if it is from the historical newspapers database
    #This doesn't always work -- b/c of a bad import on the wsj itself
    s1='ProQuest Historical Newspapers: The Wall Street Journal'
    #Check if it is from the newspaper database, old edition
    s2='Wall Street Journal(1889 - 1922);'
    #Check if it is from the newspaper database, new edition
    s3='Wall Street Journal(1923 - Current file);'
    #Fix for problem with wsj import
    s4='ProQuest Historical Newspapers:'
    #Deal with the aditional format™£
    sn1='Abstract Translate'
    sn2='Full Text Translate'
    sn3='Word count'
    # Account for other more recent articles
    sn4 = 'More like this'
    sn5 = 'Full text Translate'
    #Want to accout for the 2018 style
    snn1 = 'FULL TEXT'
    snn2 = 'DETAILS'
    #Last one
    nuformat='Full Text'
    nu = article.find(nuformat)
    nuformat2='Details\r\nTitle'
    nu2 = article.find(nuformat2)
    #needs to have nu!=-1 and snn1==-1 and sn2==-1 and sn1==-1 and snn2==-1

    f1 = article.find(s1)
    f2 = article.find(s2)
    f3 = article.find(s3)
    f4 = article.find(s4)

    fn1=article.find(sn1)
    fn2=article.find(sn2)
    fn3=article.find(sn3)
    fn4 = article.find(sn4)
    fn5 = article.find(sn5)

    fnn1=article.find(snn1)
    fnn2=article.find(snn2)

    if f1!=-1:
        #print('historical newspaper')
        #Take everything after the header
        b1=article.split(s1)
        article = b1[1]
    elif f1==-1 and f2==1:
        #print('newspaper (old)')
        #Take everything after the header
        b1=article.split(s2)
        article = b1[1]
    elif f1==-1 and f3==1:
        #print('newspaper (new)')
        #Take everything after the header
        b1=article.split(s3)
        article = b1[1]
    elif f1==-1 and f4==1:
        #print('incomplete read')
        #Take everything after the header
        b1=article.split(s4)
        article = b1[1]
    #Have to comment the whole thing, and the printing is annoying
    #else:
    #    print('No Header')
    elif fn1!=-1 and fn2!=-1 and fn3!=-1:
        #print("new")
        b1=article.split(sn2)
        #Take after full text translate
        tart=b1[1]
        tart2=tart.split(sn3)
        #take part before the word count
        article = tart2[0]
    elif fn1!=-1 and fn5!=-1 and fn4!=-1:
        #print("new (no ct)")
        b1 = article.split(sn5)
        # Take after full text translate
        tart = b1[1]
        tart2 = tart.split(sn4)
        # take part before the word count
        article = tart2[0]
    elif fnn1!=-1 and fnn2!=-1:
        #print("newest")
        b1 = article.split(snn1)
        # Take after full text
        tart = b1[1]
        tart2 = tart.split(snn2)
        # take part before the details
        article = tart2[0]
    elif nu!=-1 and nu2!=-1 and fnn1==-1 and fn2==-1 and fn1==-1 and fnn2==-1:
        #print("nu format")
        b1 = article.split(nuformat)
        # Take after full text
        tart = b1[1]
        tart2 = tart.split(nuformat2)
        # take part before the details
        article = tart2[0]

    #One thing to do before moving to lower case, remove "Credit:" and other proquest stuff
    article = re.sub(r'Credit:', ' ', article, flags=re.MULTILINE)
    article = re.sub(r'contributed to this article', ' ', article, flags=re.MULTILINE)
    article = re.sub(r'PDF GENERATED BY SEARCH\.PROQUEST\.COM', ' ', article, flags=re.MULTILINE)

    #I think there are still some upper/lower case issues -- if we don't care about proper nouns, do this for now
    article=article.lower()

    article=cleaning_code(article)

    whitelist = set('abcdefghijklmnopqrstuvwxy ABCDEFGHIJKLMNOPQRSTUVWXYZ \.')
    article = ''.join(filter(whitelist.__contains__, article))
    article = re.sub(r'\s\s', ' ', article, flags=re.MULTILINE)

    #Filter multiple periods in a row, these also help filter tables of returns
    #can't get all at once, so need to loop (deals with more than two periods in a row)
    for i in range (0,12):
        article = re.sub(r'\.\.', '.', article, flags=re.MULTILINE)
    #Still have some sentences with nothing in them -- these will get filtered out later anyway so it is okay

    #Split on periods
    sents=article.split(".")
    #Note -- this is not perfect -- sometimes commas get read as periods for the old articles

    #Create a list to put the sentences back into
    sentences=[]

    #Loop over sentences
    for sent in sents:
        #making everything lowercase to make checking typos easier
        sent=sent.lower()
        #print(sent)
        #Split words on spaces
        words=sent.split(" ")

        #Before blank strings i.e. "" were getting counted as typos, this should remove them
        words =list(filter(None, words))
        #Re-construct sentence w/o typos
        sent2 = ""

        for word in words:
            # If english, add back, otherwise add to typos
            if (word in english_words):
                # Adding an additional filter -- All words at least 3 letters
                if (len(word) > 2):
                    sent2 = sent2 + " " + word

        # Need to again remove 1 letter words -- even though they are not typos don't want them
        # Not sure why they are still showing up
        sent2 = ' '.join([w for w in sent2.split() if len(w) > 1])

        # print(sent2)
        if len(sent2) > 0:
            # Going to apply an additional filter -- to be added to the reconstructed article,
            # need at least 3 words in a sentence
            if len(sent2.split()) > 3:
                sentences.append(sent2)

    article2=' '.join(sentences)
    article2 = re.sub(r'\s\s', ' ', article2, flags=re.MULTILINE)

    #Tokenize the sentence
    word_tokens = word_tokenize(article2)
    #print(stop_words)
    filt_doc = [w for w in word_tokens] #if not w in stop_words]
    article2 = ' '.join(filt_doc)
    return article2
