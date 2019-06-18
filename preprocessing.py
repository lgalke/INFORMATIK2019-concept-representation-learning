import itertools as it
from collections import Counter, defaultdict

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


def normalize_text(text, lower=True):
    """
    Normalizes a string.
    The string is lowercased and all non-alphanumeric characters are removed.

    >>> normalize_text("already normalized")
    'already normalized'
    >>> normalize_text("This is a fancy title / with subtitle ")
    'this is a fancy title with subtitle'
    >>> normalize_text("#@$~(@ $*This has fancy \\n symbols in it \\n")
    'this has fancy symbols in it'
    >>> normalize_text("Oh no a ton of special symbols: $*#@(@()!")
    'oh no a ton of special symbols'
    >>> normalize_text("A (2009) +B (2008)")
    'a 2009 b 2008'
    >>> normalize_text("1238912839")
    '1238912839'
    >>> normalize_text("#$@(*$(@#$*(")
    ''
    >>> normalize_text("Now$ this$ =is= a $*#(ing crazy string !!@)# check")
    'now this is a ing crazy string check'
    >>> normalize_text("Also commata, and other punctuation... is not alpha-numeric")
    'also commata and other punctuation is not alphanumeric'
    >>> normalize_text(("This goes over\\n" "Two Lines"))
    'this goes over two lines'
    >>> normalize_text('')
    ''
    """
    text = str(text)
    if lower:
        text = text.lower()
    return ' '.join(filter(None, (''.join(c for c in w if c.isalnum())
                                  for w in text.split())))

class AlphaNumericTextPreprocessor(object):

    """Removes non-alphanumeric symbols such as punctuation and splits on whitespace"""

    def __init__(self, max_features=None, lowercase=True, max_length=None,
                 stop_words=None, drop_unknown=False, dtype=None):
        """Initializes configuration

        :max_features: maximum vocabulary size
        :lowercase: lowercase everything
        :max_length: maximum and globally constant length for output
        :stop_words: list of stop words to ignore

        """
        self._max_features = max_features
        self._lowercase = lowercase
        self._max_length = max_length
        # Set for faster access
        if isinstance(stop_words, str):
            if stop_words == 'english':
                self._stop_words = ENGLISH_STOP_WORDS
            else:
                raise ValueError("Stop words should be 'english', a list or set!")
        self._stop_words = set(stop_words) if stop_words else None
        self._dtype = dtype
        self._drop_unknown = drop_unknown
        if drop_unknown:
            self._offset, self.unk_idx_ = 1, None
        else:
            # Static index 1 for unk tokens
            self._offset, self.unk_idx_ = 2, 1

        self.padding_idx_ = 0  # Static index for padding

        self.vocabulary_ = None

    def _tokenize(self, raw_string):
        tokens = normalize_text(raw_string, lower=self._lowercase).split()
        if self._stop_words:
            tokens = [tok for tok in tokens if tok not in self._stop_words]
        return tokens

    def _build_vocab(self, flat_tokens):
        counter = Counter(flat_tokens)  # Counts tokens
        most_common = counter.most_common(self._max_features)  # Get most common ones
        # words, counts = list(zip(*most_common))
        vocabulary = {word:i+self._offset for i, (word, __count) in enumerate(most_common)}
        if not self._drop_unknown:
            # Map unknown tokens to index 1 per default
            vocabulary = defaultdict(lambda: 1, vocabulary)
        return vocabulary

    def _apply_vocab(self, doc):
        if self._drop_unknown:
            # Drop out-of-vocabulary tokens
            doc = [tok for tok in doc if tok in self.vocabulary_]

        # Index tokens
        doc = [self.vocabulary_[tok] for tok in doc]
        return doc

    def _pad(self, batch):
        if self._max_length is not None:
            # Use pre-defined length
            maxlen = self._max_length

            # No document should be longer than max length
            batch = [doc[:maxlen] for doc in batch]
        else:
            # Use max length of batch
            maxlen = max(map(len, batch))
            # Here, no document can exceed max length

        # Pad shorter docs with padding symbol 0
        batch = [doc + [self.padding_idx_] * (maxlen - len(doc)) for doc in batch]

        return batch

    def fit(self, raw_documents):
        """fit
        Builds a vocabulary.
        :param raw_documents: List[str]
        """
        print("Tokenizing...")
        docs = [self._tokenize(raw_doc) for raw_doc in raw_documents]
        print("Building vocabulary...")
        self.vocabulary_ = self._build_vocab(it.chain.from_iterable(docs))
        return self

    def transform(self, raw_documents):
        """transform
        Transforms raw documents into indices.

        :param raw_documents: List[str]
        """
        print("Indexing tokens...")
        docs = [self._tokenize(raw_doc) for raw_doc in raw_documents]
        docs_ix = [self._apply_vocab(doc) for doc in docs]
        docs_ix = self._pad(docs_ix)

        if self._dtype is None:
            return docs_ix

        return self._dtype(docs_ix)


    def fit_transform(self, raw_documents):
        """fit_transform
        Builds a vocabulary and transforms raw documents into indices.
        Saves one tokenization pass through the documents compared to calling
        fit and transform one after the other.

        :param raw_documents:
        """
        print("Tokenizing...")
        docs = [self._tokenize(raw_doc) for raw_doc in raw_documents]

        print("Building vocabulary...")
        self.vocabulary_ = self._build_vocab(it.chain.from_iterable(docs))

        print("Indexing tokens...")
        docs_ix = [self._apply_vocab(doc) for doc in docs]
        docs_ix = self._pad(docs_ix)

        if self._dtype is None:
            return docs_ix

        return self._dtype(docs_ix)

