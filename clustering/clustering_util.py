import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re

# GLOBALS
NUMBER_TOPICS = 10
NUMBER_WORDS = 5


def clean_data(df):
    '''
    # Remove punctuation, convert to lowercase, remove stop words, remove redacted/x'ed out terms

    Takes: df
    Returns: cleaned df
    '''
    # Remove punctuation, convert to lowercase, remove stop words, remove redacted/x'ed out terms
    stop = stopwords.words('english')
    df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'] \
        .map(lambda x: re.sub(r'[,/)(\.!?]', '', x)) \
        .map(lambda x: x.lower()) \
        .apply(lambda x: ' '.join([i for i in x.split() if i not in stop])) \
        .str.replace(r"xx+\s", "") \
        .apply(standardize_terms) \
        .apply(lambda x: ' '.join([i for i in x.split() if i not in stop]))

    return df

def create_term_dic():
    '''
    # Create term dic as combination of two popular term dicts

    Takes: None
    Returns: term dictionary
    '''
    # Remove punctuation, convert to lowercase, remove stop words, remove redacted/x'ed out terms
    dico = {}
    dico2 = open('norm_dict/emnlp_dict.txt', 'rb')
    for word in dico2:
        word = word.decode('utf8')
        word = word.split()
        dico[word[0]] = word[1]
    dico2.close()
    dico3 = open('norm_dict/typo-corpus-r1.txt', 'rb')
    for word in dico3:
        word = word.decode('utf8')
        word = word.split()
        dico[word[0]] = word[1]
    dico3.close()

    return dico


def standardize_terms(words):
    '''
    # Standardize terms

    Takes: term dictionary and words to standardize
    Returns: list of words
    '''
    dico = create_term_dic()
    list_words = words.split()
    for i in range(len(list_words)):
        if list_words[i] in dico.keys():
            list_words[i] = dico[list_words[i]]
    return ' '.join(list_words)


def print_lda_topics(model, count_vectorizer, n_top_words):
    '''
    # Helper function to help show top words per cluster

    Takes: model, count_vectorizer, top_word_count
    Returns: prints top words
    '''
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
