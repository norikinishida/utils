from collections import OrderedDict, Counter
from configparser import SafeConfigParser
import datetime
import hashlib
import json
import logging
from logging import getLogger, Formatter, StreamHandler, DEBUG
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from chainer import cuda, optimizers, Variable

import visualizers

###############################
# Logging

logger = getLogger("logger")
logger.setLevel(DEBUG)

handler = StreamHandler()
handler.setLevel(DEBUG)
handler.setFormatter(Formatter(fmt="%(message)s"))
logger.addHandler(handler)

def writelog(msgtype, text):
    """
    :type msgtype: str
    :type text: str
    :rtype: None
    """
    logger.debug("[%s] %s" % (msgtype, text))

def set_logger(filename):
    if os.path.exists(filename):
        writelog("utils.set_logger", "A file %s already exists." % filename)
        do_remove = input("[utils.set_logger] Delete the existing log file? [y/n]: ")
        if (not do_remove.lower().startswith("y")) and (not len(do_remove) == 0):
            writelog("utils.set_logger", "Done.")
            sys.exit(0)
    logging.basicConfig(level=DEBUG, format="%(message)s", filename=filename, filemode="w")

############################
# Functions/Classes for configulation

class Config(object):
    def __init__(self, path_config=None):
        self.parser = SafeConfigParser()
        self.parser.read("./config/path.ini")
        if path_config is not None:
            if not os.path.exists(path_config):
                print("Error!: path_config=%s does not exist." % path_config)
                sys.exit(-1)
            self.parser.read(path_config)

    def getpath(self, key):
        return self.str2None(json.loads(self.parser.get("path", key)))

    def getint(self, key):
        return self.parser.getint("hyperparams", key)

    def getfloat(self, key):
        return self.parser.getfloat("hyperparams", key)

    def getbool(self, key):
        return self.parser.getboolean("hyperparams", key)

    def getstr(self, key):
        return self.str2None(json.loads(self.parser.get("hyperparams", key)))

    def getlist(self, key):
        xs = json.loads(self.parser.get("hyperparams", key))
        xs = [self.str2None(x) for x in xs]
        return xs

    def getdict(self, key):
        xs  = json.loads(self.parser.get("hyperparams", key))
        for key in xs.keys():
            value = self.str2None(xs[key])
            xs[key] = value
        return xs

    def str2None(self, s):
        if isinstance(s, str) and s == "None":
            return None
        else:
            return s

def add_lines_to_configfile(path, new_lines, previous_key):
    """
    :type path: str
    :type lines: list of str
    :type previous_line: str
    :rtype: None
    """
    cur_lines = open(path).readlines()
    print(path)
    with open(path, "w") as f:
        for cur_line in cur_lines:
            cur_line = cur_line.strip()
            f.write("%s\n" % cur_line)
            print(cur_line)

            key = cur_line.split()[0]
            if key == previous_key:
                for new_line in new_lines:
                    f.write("%s\n" % new_line)
                    print(new_line)

def replace_line_in_configfile(path, new_line, target_key):
    """
    :type path: str
    :type line: str
    :type target_key: str
    :rtype: None
    """
    cur_lines = open(path).readlines()
    print(path)
    written = False
    with open(path, "w") as f:
        for cur_line in cur_lines:
            cur_line = cur_line.strip()
            if cur_line == "":
                f.write("\n")
                continue
            key = cur_line.split()[0]
            if key != target_key:
                f.write("%s\n" % cur_line)
                print(cur_line)
            else:
                assert not written
                f.write("%s\n" % new_line)
                print(new_line)
                written = True

def dump_hyperparams_summary(path_in, path_out, exception_names):
    """
    :type path_in: str
    :type path_out: str
    :type exception_names: list of str
    """
    result = [] # list of {str: str}

    # Make a list of target filenames
    filenames = os.listdir(path_in)
    for filename in exception_names:
        filenames.remove(filename)
    filenames = [filename for filename in filenames
                 if filename.endswith(".ini")]
    filenames.sort()

    for filename in filenames:
        # Prepare a Config instance
        path_config = os.path.join(path_in, filename)
        config = Config(path_config)
        # Read key-value pairs
        keyval = OrderedDict()
        keyval["name"] = filename
        assert "hyperparams" in config.parser
        for key in config.parser["hyperparams"].keys():
            keyval[key] = config.parser["hyperparams"][key]
        result.append(keyval)

    df = pd.DataFrame(result)
    print(df)
    df.to_csv(path_out, index=False)

############################
# Functions/Classes for general purpose

def get_basename_without_ext(path):
    """
    :type path: str
    :rtype: str
    """
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]

def get_current_time():
    """
    :rtype: str
    """
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def get_random_english_word():
    """
    :rtype: word
    """
    path = os.path.join(os.path.dirname(__file__), "englishwords.txt")
    words = read_lines(path)
    word = np.random.choice(words)
    return word

def hash_string(text):
    """
    :type text: str
    :rtype: int
    """
    h = hashlib.sha256(text.encode()).hexdigest()
    i = str(int(h, 16))
    return int(i[:8]) # to limit the value between 0 and 2***32-1

############################
# Functions/Classes for general IO

def mkdir(path, newdir=None):
    """
    :type path: str
    :type newdir: None, or str
    :rtype: None
    """
    if newdir is None:
        target = path
    else:
        target = os.path.join(path, newdir)
    if not os.path.exists(target):
        os.makedirs(target)
        writelog("utils.mkdir", "Created a directory=%s" % target)

def read_vocab(path):
    """
    :type path: str
    :rtype: dictionary of {str: int}
    """
    begin_time = time.time()
    writelog("utils.read_vocab", "Loading a vocabulary from %s" % path)
    vocab = OrderedDict()
    for line in open(path):
        word, word_id, freq = line.strip().split("\t")
        vocab[word] = int(word_id)
    end_time = time.time()
    writelog("utils.read_vocab", "Loaded. %f [sec.]" % (end_time - begin_time))
    writelog("utils.read_vocab", "Vocabulary size=%d" % len(vocab))
    return vocab

def read_lines(path, process=lambda line: line):
    """
    :type path: str
    :type process: function
    :rtype: list of Any
    """
    lines = []
    for line in open(path):
        line = line.strip()
        line = process(line)
        lines.append(line)
    return lines

def write_lines(path, lines, process=lambda line: line):
    """
    :type path: str
    :type lines: list of Any
    :type process: function
    :rtype: None
    """
    with open(path, "w") as f:
        for line in lines:
            line = process(line)
            f.write("%s\n" % line)

def read_vectors(path):
    """
    :type path: str
    :rtype: numpy.ndarray(shape=(N,dim), dtype=float)
    """
    vectors = []
    for line in open(path):
        vector = line.strip().split()
        vector = [float(x) for x in vector]
        vectors.append(vector)
    vectors = np.asarray(vectors)
    return vectors

def write_vectors(path, vectors):
    """
    :type path: str
    :type vectors: numpy.ndarray(shape=(N,dim), dtype=float)
    :rtype: None
    """
    with open(path, "w") as f:
        for i in range(len(vectors)):
            vector = vectors[i]
            vector = [str(x) for x in vector]
            vector = " ".join(vector)
            f.write("%s\n" % vector)

def read_conll(path, keys):
    """
    :type path: str
    :type keys: list of str
    :rtype: list of list of {str: str}

    CoNLL-X: ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
    CoNLL-U: ID FORM LEMMA UPOS    XPOS   FEATS HEAD DEPREL DEPS  MISC
    """
    sentences = []

    n_items = len(keys)

    sentence = []
    for line in open(path):
        line = line.strip()
        if not line:
            if len(sentence) != 0:
                sentences.append(sentence)
                sentence = []
            continue
        items = line.split("\t")
        assert len(items) == n_items
        conll_line = {key:item for key,item in zip(keys, items)}
        sentence.append(conll_line)
    if sentence:
        sentences.append(sentence)
    return sentences

def write_conll(path, sentences):
    """
    :type path: str
    :type sentences: list of list of {str: str}
    :rtype: None
    """
    with open(path, "w") as f:
        for sentence in sentences:
            for conll_line in sentence:
                items = [conll_line[key] for key in conll_line.keys()]
                f.write("\t".join(items) + "\n")
            f.write("\n")

def convert_conll_to_linebyline_format(path_conll, keys, ID, FORM, POSTAG, HEAD, DEPREL):
    """
    :type path_conll: str
    :type keys: list of str
    :type ID: str
    :type FORM: str
    :type POSTAG: str
    :type HEAD: str
    :type DEPREL: str
    :rtype: list of str, list of str, list of (int, int, str)
    """
    assert ID in keys
    assert FORM in keys
    assert POSTAG in keys
    assert HEAD in keys
    assert DEPREL in keys

    batch_tokens = []
    batch_postags = []
    batch_arcs = []

    n_items = len(keys)

    # Init
    tokens = []
    postags = []
    arcs = []

    for line in open(path_conll):
        line = line.strip()
        if line == "":
            if len(tokens) == 0:
                continue
            batch_tokens.append(tokens)
            batch_postags.append(postags)
            batch_arcs.append(arcs)
            # Init
            tokens = []
            postags = []
            arcs = []
        else:
            items = line.split("\t")
            assert len(items) == n_items
            conll_line = {key:item for key,item in zip(keys, items)}

            dep_index = int(conll_line[ID])
            token = conll_line[FORM]
            postag = conll_line[POSTAG]
            head_index = int(conll_line[HEAD])
            label = conll_line[DEPREL]

            tokens.append(token)
            postags.append(postag)
            arcs.append((head_index, dep_index, label))

    if len(tokens) != 0:
        batch_tokens.append(tokens)
        batch_postags.append(postags)
        batch_arcs.append(arcs)

    return batch_tokens, batch_postags, batch_arcs

def extract_values_with_regex(filepath, regex, names):
    """
    :type filepath: str
    :type regex: str
    :type names: list of str
    :rtype: dictionary of {str: list of str}
    """
    re_comp = re.compile(regex, re.I)
    values = {name: [] for name in names}
    for line in open(filepath):
        line = line.strip()
        match = re_comp.findall(line)
        if len(match) > 0:
            match = match[0]
            if not isinstance(match, tuple):
                match = (match,)
            assert len(match) == len(names)
            for index in range(len(names)):
                values[names[index]].append(match[index])
    return values

def print_list(lst):
    """
    :type lst: list of Any
    :rtype: None
    """
    for x in lst:
        print(x)

def print_dict(dictionary):
    """
    :type dictionary: {Any: Any}
    :rtype: None
    """
    for key in dictionary.keys():
        print("%s: %s" % (key, dictionary[key]))

############################
# Functions/Classes for manipulating lists/arrays/dictionary

def filter_by_condition(xs, ys, condition_function):
    """
    :type xs: list of list of Any
    :type ys: list of Any
    :type condition_function: function
    :rtype: list of Any
    """
    assert len(xs) == len(ys)
    indices = [i for i, x in enumerate(xs) if condition_function(x)]
    zs = [ys[i] for i in indices]
    return zs

def flatten_lists(list_of_lists):
    """
    :type list_of_lists: list of list of T
    :rtype: list of T
    """
    return [elem for lst in list_of_lists for elem in lst]

def compare_dictionary_keys(dict1, dict2):
    """
    :type dict1: {Any: Any}
    :type dict2: {Any: Any}
    :rtype: bool
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    if len(keys1 & keys2) == len(keys1) == len(keys2):
        return True
    else:
        False

############################
# Functions/Classes for distance calculation

def levenshtein_distance(seq1, seq2):
    """
    :type seq1: list of Any
    :type seq2: list of Any
    :rtype: float
    """
    length1 = len(seq1)
    length2 = len(seq2)

    table = np.zeros((length1 + 1, length2 + 1))

    # Base case
    for i1 in range(length1 + 1):
        table[i1, 0] = i1
    for i2 in range(length2 + 1):
        table[0, i2] = i2

    # General case
    for i1 in range(1, length1 + 1):
        for i2 in range(1, length2 + 1):
            if seq1[i1 - 1] == seq2[i2 - 1]:
                cost = 0.0
            else:
                cost = 1.0
            table[i1, i2] = min(table[i1-1, i2] + 1.0, # Insertion
                                table[i1, i2-1] + 1.0, # Deletion
                                table[i1-1, i2-1] + cost) # Replacement/Nothing

    return table[length1, length2]

############################
# Functions/Classes for data processing

class DataBatch(object):

    def __init__(self, **kargs):
        self._attr_names = []
        length = None
        for key, value in kargs.items():
            setattr(self, key, value)
            self._attr_names.append(key)
            # Check
            if length is None:
                length = len(value)
            else:
                assert length == len(value)

    def __len__(self):
        return len(getattr(self, self._attr_names[0]))

class DataPool(object):

    def __init__(self, paths, processes=None, pool_size=1000000):
        """
        :type paths: list of str
        :type processes: list of function
        :type pool_size: int
        """
        self.paths = paths

        if processes is None:
            self.processes = [lambda l: l for _ in range(self.paths)]
        else:
            assert len(processes) == len(self.paths)
            self.processes = processes

        self._pool_attr_names = ["pool_%d" % path_i for path_i in range(len(self.paths))]

        # Count the number of lines in the text files
        writelog("utils.DataPool", "Counting the number of lines in the text files ...")
        self._n_lines = None
        for path in self.paths:
            # Count
            count = 0
            for _ in open(path):
                count += 1
            # Check
            if self._n_lines is None:
                self._n_lines = count
            else:
                assert count == self._n_lines

        # Set the pool size
        self.pool_size = min(pool_size, self._n_lines)

        # Initialize the iterator
        self._current_iterator = self._get_init_iterator()
        self._line_i = 0

        # Create the pools
        for pool_attr_name in self._pool_attr_names:
            empty_pool = np.zeros((self.pool_size), dtype="O")
            setattr(self, pool_attr_name, empty_pool)
        self._fill_pools(indices=None)

    def __len__(self):
        return self._n_lines

    def __iter__(self):
        """
        :rtype: list of Any
        """
        for tpl in self._get_init_iterator():
            lst = self._process(tpl)
            yield lst

    def get_instances(self, batch_size):
        """
        :type batch_size: int
        :rtype: list of numpy.ndarray(shape=(batch_size,), dtype="O")
        """
        indices = np.random.choice(self.pool_size, size=batch_size) # NOTE that ``replace'' is True.
        output = [getattr(self, pool_attr_name)[indices] for pool_attr_name in self._pool_attr_names]
        self._fill_pools(indices=indices)
        return output

    def _get_init_iterator(self):
        return zip(*[open(path) for path in self.paths])

    def _process(self, tpl):
        """
        :type tpl: tuple of str
        :rtype: list of Any
        """
        return [process(line.strip()) for line, process in zip(tpl, self.processes)]

    def _read_line(self):
        """
        :rtype: list of Any
        """
        if self._line_i >= self._n_lines:
            self._current_iterator = self._get_init_iterator()
            self._line_i = 0

        tpl = next(self._current_iterator)
        lst = self._process(tpl)
        self._line_i += 1
        return lst

    def _fill_pools(self, indices=None):
        """
        :type indices: list of int
        :rtype: None
        """
        if indices is None:
            indices = range(self.pool_size)

        for index in indices:
            # Read
            lst = self._read_line()
            # Assign
            for pool_attr_name, line in zip(self._pool_attr_names, lst):
                getattr(self, pool_attr_name)[index] = line

    def get_random_instances(self, n_instances):
        """
        :type n_instances: int
        :rtype: list of list of Any
        """
        output = []
        line_indices = np.random.choice(self._n_lines, size=n_instances, replace=False)
        line_i = 0
        for tpl in self._get_init_iterator():
            if line_i in line_indices:
                lst = self._process(tpl)
                output.append(lst)
            line_i += 1
        return output

############################
# Functions/Classes for basic feature vectors

def make_multihot_vectors(dim, fire):
    """
    :type dim: int
    :type fire: list of list of int
    :rtype: numpy.ndarray(shape=(N, dim), dtype=np.float32)
    """
    n_instances = len(fire)
    vectors = np.zeros((n_instances, dim), dtype=np.float32)
    for instance_i in range(n_instances):
        vectors[instance_i, fire[instance_i]] = 1.0
    return vectors

class BoW(object):

    def __init__(self, documents, tfidf):
        """
        :type documents: list of list of str
        :type tfidf: bool
        :rtype: None
        """
        self.tfidf = tfidf

        if not tfidf:
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = TfidfVectorizer()

        self.vectorizer.fit_transform([" ".join(d) for d in documents])

        vocab_words = self.vectorizer.get_feature_names()
        self.vocab = {w:i for i,w in enumerate(vocab_words)}

    def forward(self, documents):
        """
        :type documents: list of list of str
        :rtype: numpy.ndarray(shape=(N,|V|), dtype=np.float32)
        """
        X = self.vectorizer.transform([" ".join(d) for d in documents])
        return X.toarray().astype(np.float32)

############################
# Functions/Classes for pre-trained word embeddings

def read_word_embedding_matrix(path, dim, vocab, scale):
    """
    :type path: str
    :type dim: int
    :type vocab: dictionary of {str: int}
    :type scale: float
    :rtype: numpy.ndarray(shape=(vocab_size, dim), dtype=np.float32)
    """
    word2vec = read_word2vec(path, dim)
    W = convert_word2vec_to_weight_matrix(vocab, word2vec, dim, scale)
    return W

def read_word2vec(path, dim):
    """
    :type path: str
    :type dim: int
    :rtype: {str: numpy.ndarray(shape=(dim,), dtype=np.float32)}
    """
    writelog("utils.read_word2vec", "Loading pretrained word vectors from %s ..." % path)
    begin_time = time.time()

    word2vec = {}
    with open(path) as f:
        for line_i, line in enumerate(f):
            items = line.strip().split()
            if len(items[1:]) != dim:
                writelog("utils.read_word2vec", "dim %d(actual) != %d(expected), skipped %d-th line=%s..." % \
                        (len(items[1:]), dim, line_i+1, ",".join(items[:10])))
                continue
            word2vec[items[0]] = np.asarray([float(x) for x in items[1:]])

    end_time = time.time()
    writelog("utils.read_word2vec", "Loaded. %f [sec.]" % (end_time - begin_time))
    writelog("utils.read_word2vec", "Vocabulary size=%d" % len(word2vec))
    return word2vec

def convert_word2vec_to_weight_matrix(vocab, word2vec, dim, scale):
    """
    :type vocab: {str -> int}
    :type word2vec: {str -> numpy.ndarray(shape=(dim,), dtype=np.float32)}
    :type dim: int
    :type scale: float
    :rtype: numpy.ndarray(shape=(vocab_size, dim), dtype=np.float32)
    """
    writelog("utils.convert_word2vec_to_weight_matrix", "Converting ...")
    begin_time = time.time()

    task_vocab = list(vocab.keys())
    word2vec_vocab = list(word2vec.keys())
    shared_vocab = set(task_vocab) & set(word2vec_vocab)
    writelog("utils.convert_word2vec_to_weight_matrix", "Vocabulary size (task)=%d" % len(task_vocab))
    writelog("utils.convert_word2vec_to_weight_matrix", "Vocabulary size (word2vec)=%d" % len(word2vec_vocab))
    writelog("utils.convert_word2vec_to_weight_matrix", "Vocabulary size (shared)=%d (|shared|/|task|=%d/%d=%.2f%%)" % \
            (len(shared_vocab), len(shared_vocab), len(task_vocab),
                float(len(shared_vocab))/len(task_vocab)*100.0))

    # NOTE: If we fix the word vectors, we should use the same random seed for initializing the out-of-vocabulary words.
    W = np.random.RandomState(1234).uniform(-scale, scale, (len(task_vocab), dim)).astype(np.float32)
    for w in shared_vocab:
        W[vocab[w], :] = word2vec[w]

    end_time = time.time()
    writelog("utils.convert_word2vec_to_weight_matrix", "Converted. %f [sec.]" % (end_time - begin_time))
    return W

############################
# Functions/Classes for neural network models (using Chainer)

def transform_words(xs):
    """
    :type xs: list of list(len=L) of int
    :rtype: list of Variable(shape=(L,), dtype=np.int32)
    """
    xs = [np.asarray(x, dtype=np.int32) for x in xs]
    xs = [Variable(cuda.cupy.asarray(x)) for x in xs]
    return xs

def padding(xs, head, with_mask):
    """
    :type xs: list of list of int
    :type head: bool
    :type with_mask: bool
    :rtype: numpy.ndarray(shape=(N, max_length)), numpy.ndarray(shape(N, max_length))
    """
    N = len(xs)
    max_length = max([len(x) for x in xs])
    ys = np.zeros((N, max_length), dtype=np.int32)
    if head:
        for i in range(N):
            l = len(xs[i])
            ys[i, 0:l] = xs[i]
            ys[i, l:] = -1
    else:
        for i in range(N):
            l = len(xs[i])
            ys[i, 0:max_length-l] = -1
            ys[i, max_length-l:] = xs[i]
    if with_mask:
        ms = np.greater(ys, -1).astype(np.float32)
        return ys, ms
    else:
        return ys

def convert_ndarray_to_variable(xs, seq):
    """
    :type xs: numpy.ndarray(shape=(N, L))
    :type seq: bool
    :rtype: list(len=L) of Variable(shape=(N,)), or Variable(shape=(N, L))
    """
    if seq:
        return [Variable(cuda.cupy.asarray(xs[:,j]))
                for j in range(xs.shape[1])]
    else:
        return Variable(cuda.cupy.asarray(xs))

def get_optimizer(name="smorms3"):
    """
    :type name: str
    :rtype: chainer.Optimizer
    """
    if name == "adadelta":
        opt = optimizers.AdaDelta()
    elif name == "adagrad":
        opt = optimizers.AdaGrad()
    elif name == "adam":
        opt = optimizers.Adam()
    elif name == "rmsprop":
        opt = optimizers.RMSprop()
    elif name == "smorms3":
        opt = optimizers.SMORMS3()
    else:
        raise ValueError("Unknown optimizer_name=%s" % name)
    return opt

############################
# Functions/Classes for machine learning training

class BestScoreHolder(object):

    def __init__(self, scale=1.0, higher_is_better=True):
        self.scale = scale
        self.higher_is_better = higher_is_better

        if higher_is_better:
            self.comparison_function = lambda best, cur: best < cur
        else:
            self.comparison_function = lambda best, cur: best > cur

        if higher_is_better:
            self.best_score = -np.inf
        else:
            self.best_score = np.inf
        self.best_step = 0
        self.patience = 0

    def init(self):
        if self.higher_is_better:
            self.best_score = -np.inf
        else:
            self.best_score = np.inf
        self.best_step = 0
        self.patience = 0

    def compare_scores(self, score, step):
        if self.comparison_function(self.best_score, score):
            # Update the score
            writelog("utils.BestScoreHolder", "(best_score=%.02f, best_step=%d, patience=%d) -> (%.02f, %d, %d)" % \
                    (self.best_score * self.scale, self.best_step, self.patience,
                     score * self.scale, step, 0))
            self.best_score = score
            self.best_step = step
            self.patience = 0
            return True
        else:
            # Increment the patience
            writelog("utils.BestScoreHolder", "(best_score=%.02f, best_step=%d, patience=%d) -> (%.02f, %d, %d)" % \
                    (self.best_score * self.scale, self.best_step, self.patience,
                     self.best_score * self.scale, self.best_step, self.patience+1))
            self.patience += 1
            return False

    def ask_finishing(self, max_patience):
        if self.patience >= max_patience:
            return True
        else:
            return False

############################
# Functions/Classes for analysis

def get_word_counter(path=None, lines=None):
    """
    :type path: str
    :type lines: list of list of str
    :type process: function
    """
    counter = Counter()

    if path is not None:
        assert lines is None
        for line in open(path):
            tokens = line.strip().split()
            counter.update(tokens)
    elif lines is not None:
        assert path is None
        for tokens in lines:
            counter.update(tokens)
    else:
        raise ValueError("Both ``path'' and ``lines'' are None.")

    return counter

def calc_word_stats(path_dir, top_k, process=lambda line: line.split()):
    """
    :type path_dir: str
    :type top_k: int
    :type process: function
    :rtype: None
    """
    filenames = os.listdir(path_dir)

    stopwords = read_lines(os.path.join(os.path.dirname(__file__), "stopwords.txt"))

    counter = Counter()
    for filename in filenames:
        c = get_word_counter(path=os.path.join(path_dir, filename))
        counter.update(c)
    counter = counter.most_common()
    counter = counter[:1000]

    # Filtering stopwords
    counter = [(w, freq) for w, freq in counter if not w in stopwords]

    for k in range(min(top_k, len(counter))):
        w, freq = counter[k]
        writelog("utils.check_word_stats", "word=%s, frequency=%d" % (w, freq))

def calc_score_stats(filepaths, regex, names):
    """
    :type filepaths: list of str
    :type regex: str
    :type names: list of str
    :rtype: Pandas.DataFrame
    """
    data = {name: [] for name in names} # {str: list of float}
    for filepath in filepaths:
        scores = extract_values_with_regex(filepath, regex, names) # {str: list of str}
        for name in names:
            assert len(scores[name]) == 1
            score = float(scores[name][0])
            data[name].append(score)

    for name in names:
        data[name] = np.asarray(data[name])

    df_data = OrderedDict()
    def _calc_stats(xs):
        return xs.tolist() + [np.mean(xs), np.std(xs), np.max(xs), np.min(xs)]
    for name in names:
        df_data[name] = _calc_stats(data[name])
    df = pd.DataFrame(df_data, index=[os.path.basename(filepath) for filepath in filepaths] + ["mean", "std", "max", "min"])
    pd.options.display.float_format = "{:,.1f}".format
    return df

def plot_given_files(
        filepaths, regex,
        xticks, xlabel, ylabels,
        legend_names, legend_anchor, legend_location,
        marker="o", linestyle="-", markersize=10,
        fontsize=30,
        savepaths=None, figsize=(8,6), dpi=100):
    """
    :type filepaths: list of str
    :type regex: str
    :type xticks: list of str
    :type xlabel: str
    :type ylabels: list of str
    :type legend_names: list of str
    :type legend_anchor: (int, int)
    :type legend_location: str
    :type marker: str
    :type linestyle: str
    :type markersize: int
    :type fontsize: int
    :type savepaths: list of str
    :type figsize: (int, int)
    :type dpi: int
    :rtype: None
    """
    assert len(filepaths) == len(legend_names)

    # Extraction
    data = {ylabel: [] for ylabel in ylabels} # {str: list of list of float}
    for filepath in filepaths:
        scores = extract_values_with_regex(filepath, regex, ylabels) # {str: list of str}
        for ylabel in ylabels:
            data[ylabel].append([float(x) for x in scores[ylabel]])

    if savepaths is None:
        savepaths = [None for _ in range(len(ylabels))]

    for ylabel, savepath in zip(ylabels, savepaths):
        visualizers.plot(
                    list_ys=data[ylabel], list_xs=None,
                    xticks=xticks, xlabel=xlabel, ylabel=ylabel,
                    legend_names=legend_names,
                    legend_anchor=legend_anchor, legend_location=legend_location,
                    marker=marker, linestyle=linestyle, markersize=markersize,
                    fontsize=fontsize,
                    savepath=savepath, figsize=figsize, dpi=dpi)


