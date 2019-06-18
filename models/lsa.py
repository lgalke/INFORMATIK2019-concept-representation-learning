from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin

def class_prototypes(x_docs, y_labels):
    """
    Compute a prototype for each label, composed as the average of its
    document's representation.
    :x_docs: array-like or sparse matrix of shape (n_docs, n_features)
    :y_labels: array-like or sparse matrix of shape (n_docs, n_classes)
    :returns: class prototypes of shape (n_classes, n_features)
    """
    return normalize(y_labels.T, norm='l1') @ x_docs


class LSAEmbedding(BaseEstimator, TransformerMixin):

    """ Embedding based on latent semantic analysis.
    Can be used to obtain a node embedding based on textual node attributes
    """
    def __init__(self,
                 n_components=100,
                 max_features=50000,
                 stop_words='english'):
        self.tfidf = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
        self.svd = TruncatedSVD(n_components=n_components)

    def fit(self, raw_documents):
        """ Fits LSA to data """
        X = self.tfidf.fit_transform(raw_documents)
        self.svd.fit(X)
        return self

    def transform(self, raw_documents):
        """ Transforms raw content to LSA representation"""
        return self.svd.transform(self.tfidf.transform(raw_documents))

    def fit_transform(self, raw_documents):
        return self.svd.fit_transform(self.tfidf.fit_transform(raw_documents))
