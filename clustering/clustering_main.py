import make_datasets as md
import clustering_util as util
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pyLDAvis import sklearn as sklearn_lda
import pyLDAvis

if __name__ == "__main__":
    # get full dataset
    df = md.clean_raw_file(pd.read_csv(md.RAW_FILE))

    #clean up columns
    new_names = [n.lower().replace(" ", "_").replace("?", "") for n in df.columns]
    df.columns = new_names

    # simplify dataframe
    df2 = df2 = df[['complaint_id', 'consumer_complaint_narrative']].copy()

    # clean text data
    clean_df = util.clean_data(df2)

    # create count_vector
    count_vectorizer = CountVectorizer(stop_words='english')

    # create count_data
    count_data = count_vectorizer.fit_transform(clean_df['consumer_complaint_narrative'])

    # run LDA model - number_topics is the number of clusters
    lda = LDA(n_components=util.NUMBER_TOPICS, n_jobs=-1)

    # fit the model
    lda.fit(count_data)

    # print the top topics per cluster
    util.print_lda_topics(lda, count_vectorizer, util.NUMBER_WORDS)

    # output visualizations
    pyLDAvis.enable_notebook()
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
    pyLDAvis.display(LDAvis_prepared)
