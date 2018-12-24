from collections import OrderedDict
from configparser import SafeConfigParser
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

############################
# Functions/Classes for general purpose

def get_basename_without_ext(path):
    """
    :type path: str
    :rtype: str
    """
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]

def get_random_english_word():
    """
    :rtype: str
    """
    path = os.path.join(os.path.dirname(__file__), "./englishwords.txt")
    words = read_lines(path)
    return np.random.choice(words)

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
    writelog("utils.read_vocab", "Loading a vocabulary from %s" % path)
    vocab = OrderedDict()
    for line in open(path):
        word, word_id, freq = line.strip().split("\t")
        vocab[word] = int(word_id)
    writelog("utils.read_vocab", "Loaded the vocabulary.")
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

def append_lines(path, lines, process=lambda line: line):
    """
    :type path: str
    :type lines: list of Any
    :type process: function
    :rtype: None
    """
    with open(path, "a") as f:
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

def read_conll(path):
    """
    :type path: str
    :rtype: list of list of str
    """
    sentences = []

    sentence = []
    for line in open(path):
        line = line.strip()
        if not line:
            if len(sentence) != 0:
                sentences.append(sentence)
                sentence = []
            continue
        items = line.split("\t")
        sentence.append(items)
    if sentence:
        sentences.append(sentence)
    return sentences

def write_conll(path, sentences):
    """
    :type path: str
    :type sentences: list of list of str
    :rtype: None
    """
    with open(path, "w") as f:
        for sentence in sentences:
            for items in sentence:
                f.write("\t".join(items) + "\n")
            f.write("\n")

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
# Functions/Classes for analysis

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

############################
# Functions/Classes for loading pre-trained word vectors

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
    :rtype: dictionary of {str: numpy.ndarray(shape=(dim,), dtype=np.float32)}
    """
    writelog("utils.read_word2vec", "Loading ...")
    start_time = time.time()

    word2vec = {}
    with open(path) as f:
        for line_i, line in enumerate(f):
            items = line.strip().split()
            if len(items[1:]) != dim:
                writelog("utils.read_word2vec", "dim %d(actual) != %d(expected), skipped line %d" % \
                    (len(items[1:]), dim, line_i+1))
                continue
            word2vec[items[0]] = np.asarray([float(x) for x in items[1:]])

    writelog("utils.read_word2vec", "Loaded. %f [sec.]" % (time.time() - start_time))
    return word2vec

def convert_word2vec_to_weight_matrix(vocab, word2vec, dim, scale):
    """
    :type vocab: dictionary of {word(str) -> ID(int)}
    :type word2vec: dictionary of {word(str) -> vector(numpy.ndarray(float))}
    :type dim: int
    :type scale: float
    :rtype: numpy.ndarray of shape (vocab_size, dim) and dtype=np.float32
    """
    writelog("utils.convert_word2vec_to_weight_matrix", "Converting ...")
    start_time = time.time()

    task_vocab = list(vocab.keys())
    word2vec_vocab = list(word2vec.keys())
    common_vocab = set(task_vocab) & set(word2vec_vocab)
    writelog("utils.convert_word2vec_to_weight_matrix", "Vocabulary in the task=%d" % len(task_vocab))
    writelog("utils.convert_word2vec_to_weight_matrix", "Vocabulary in the pre-trained embeddings=%d" % len(word2vec_vocab))
    writelog("utils.convert_word2vec_to_weight_matrix", "# of pre-trained word types in the task=%d (%d/%d = %.2f%%)" % \
            (len(common_vocab), len(common_vocab), len(task_vocab),
                float(len(common_vocab))/len(task_vocab)*100.0))

    # NOTE: If we fix the word vectors, we should use the same random seed for initializing the out-of-vocabulary words.
    W = np.random.RandomState(1234).uniform(-scale, scale, (len(task_vocab), dim)).astype(np.float32)
    for w in common_vocab:
        W[vocab[w], :] = word2vec[w]

    writelog("utils.convert_word2vec_to_weight_matrix", "Converted. %f [sec.]" % (time.time() - start_time))
    return W

############################
# Functions/Classes for bag-of-word models

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
# Functions/Classes for neural network models (using Chainer)

def transform_words(xs):
    """
    :type xs: list of list(len=L) of int
    :rtype: list of Variable(shape=(L,))
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
# Functions/Classes for model training

class BestScoreHolder(object):

    def __init__(self, scale=1.0):
        self.best_score = -np.inf
        self.best_step = 0
        self.patience = 0
        self.scale = scale

    def init(self):
        self.best_score = -np.inf
        self.best_step = 0
        self.patience = 0

    def compare_scores(self, score, step):
        if self.best_score < score:
            # Update the score
            writelog("BestScoreHolder", "(best_score=%.02f, best_step=%d, patience=%d) => (%.02f, %d, %d)" % \
                    (self.best_score * self.scale, self.best_step, self.patience,
                     score * self.scale, step, 0))
            self.best_score = score
            self.best_step = step
            self.patience = 0
            return True
        else:
            # Increment the patience
            writelog("BestScoreHolder", "(best_score=%.02f, best_step=%d, patience=%d) => (%.02f, %d, %d)" % \
                    (self.best_score * self.scale, self.best_step, self.patience,
                     self.best_score * self.scale, self.best_step, self.patience+1))
            self.patience += 1
            return False

    def ask_finishing(self, max_patience):
        if self.patience >= max_patience:
            return True
        else:
            return False

