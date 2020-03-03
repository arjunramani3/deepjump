#eda1.py
from nltk.corpus import wordnet
from random import sample
from naive_bayes import load_eng_words, load_labels


#takes in an article and outputs a new article with synonyms replaced for a specified number of words
def similar_augment(article, replace_words):
    """function similar_augment creates a new article from an existing article with synonym replacement
        @param article (list of strings): a list of words specifying an article 
        @param replac_words (int): the number of words in the article to replace (must be less than len(article))
        @return new_article (list of strings): a list of words specifying the new article"""
    total_words = len(article)
    if replace_words >= totalwords: replace_words = int(total_words)/10 #correct case where replace_words is too big
    to_replace = sample(range(total_words), replace_words)
    new_article = article
    for i in to_replace:
        word = article[i]
        new_word = wordnet.synsets(word)[0].lemmas()[0].name()
        new_article[i] = new_word
    return new_article




#read in articles using import_article and labels with load_labels. Take first nwords of each article
#and create dataframe with columns Date, Words (the words in the article), [labels] where [labels] is 
#the set of labels kept from load_labels. created an augmented version of each article. Write both to the
#WSJ_agument_txt directory
def create_augmented(narts=5, nwords = 100, replace_words):
    """load_articles function
    @param narts (int): the number of articles to store in the labeled dataframe
    @param nwords (int): the number of words to keep in each article
    @param replace_words (int): the number of words to replace with synonyms in each article
    @return labeled_articles (DataFrame): a dataframe with aritcle clippings and associated labels"""
    english_words = load_eng_words()
    stop_words = set(stopwords.words('english'))
    labels = load_labels()
    #print(labels.head())
    articles = pd.DataFrame(np.zeros((narts*2, 3)), columns = ['Date', 'Words', 'Version'])
    for i, art in enumerate(os.listdir('./WSJ_txt')):
        if i > narts: break
        rawart=import_article(art,english_words,stop_words)
        #print(len(rawart.split(" ")))
        firstn=rawart.split(" ")[0:nwords]
        firstn = " ".join(firstn) #if our input is a text with spaces
        #print(firstn)
        slug = art.split('.')[0]
        articles.loc[i] = slug, firstn, 1
        new_art = similar_augment(firstn, replace_words)
        articles.loc[i+1] = slug, new_art, 2
    #print(labels.head())
    articles['Date'] = articles['Date'].str.replace('_', '/')
    print(articles['Date'])
    articles['Date'] = pd.to_datetime(articles['Date'], errors='coerce', format='%Y/%m/%d') 
    labeled_articles = articles.merge(labels, left_on = 'Date', right_on = 'Date')
    print(len(labeled_articles))
    return labeled_articles