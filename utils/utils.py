from collections import OrderedDict, Counter
from configparser import SafeConfigParser
import datetime
import hashlib
import io
import json
import jsonlines
import logging
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import gensim
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold
from chainer import cuda, Variable
import pyprind
import pyhocon

###############################
# Logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger()


def writelog(text, error=False):
    """
    Parameters
    ----------
    text: str
    error: bool
    """
    if error:
        logger.error("%s" % text)
    else:
        logger.info("%s" % text)


def set_logger(filename, overwrite=False):
    """
    Parameters
    ----------
    filename: str
    overwrite: bool, default False
    """
    if os.path.exists(filename) and not overwrite:
        print("%s already exists." % filename)
        do_remove = input("Delete the existing log file? [y/n]: ")
        if (not do_remove.lower().startswith("y")) and (not len(do_remove) == 0):
            print("Done.")
            sys.exit(0)
    logger.addHandler(logging.FileHandler(filename, "w"))


############################
# Configulation


class Config(object):
    def __init__(self, path_config=None):
        """
        Parameters
        ----------
        path_config: str or None, default None
        """
        self.parser = SafeConfigParser()
        self.parser.read("./config/path.ini")
        if path_config is not None:
            if not os.path.exists(path_config):
                writelog("Error!: path_config=%s does not exist." % path_config, error=True)
                sys.exit(-1)
            self.parser.read(path_config)

    def getpath(self, key):
        """
        Parameters
        ----------
        key: str

        Returns
        -------
        str
        """
        return self.str2None(json.loads(self.parser.get("path", key)))

    def getint(self, key):
        """
        Parameters
        ----------
        key: str

        Returns
        -------
        int
        """
        return self.parser.getint("hyperparams", key)

    def getfloat(self, key):
        """
        Parameters
        ----------
        key: str

        Returns
        -------
        float
        """
        return self.parser.getfloat("hyperparams", key)

    def getbool(self, key):
        """
        Parameters
        ----------
        key: str

        Returns
        -------
        bool
        """
        return self.parser.getboolean("hyperparams", key)

    def getstr(self, key):
        """
        Parameters
        ----------
        key: str

        Returns
        -------
        str
        """
        return self.str2None(json.loads(self.parser.get("hyperparams", key)))

    def getlist(self, key):
        """
        Parameters
        ----------
        key: str

        Returns
        -------
        list
        """
        xs = json.loads(self.parser.get("hyperparams", key))
        xs = [self.str2None(x) for x in xs]
        return xs

    def getdict(self, key):
        """
        Parameters
        ----------
        key: str

        Returns
        -------
        dict
        """
        xs  = json.loads(self.parser.get("hyperparams", key))
        for key in xs.keys():
            value = self.str2None(xs[key])
            xs[key] = value
        return xs

    def str2None(self, s):
        """
        Parameters
        ----------
        s: Any

        Returns
        -------
        Any
        """
        if isinstance(s, str) and s == "None":
            return None
        else:
            return s

    def show(self, target_section=None):
        """
        Parameters
        ----------
        target_section: str or None, default None
        """
        for section in self.parser.keys():
            if (target_section is None) or section == target_section:
                for key, value in self.parser[section].items():
                    writelog("%s = %s" % (key, value))


def add_lines_to_configfile(path, new_lines, previous_key):
    """
    Parameters
    ----------
    path: str
    lines: list[str]
    previous_line: str
    """
    cur_lines = open(path).readlines()
    print(path)
    with open(path, "w") as f:
        for cur_line in cur_lines:
            cur_line = cur_line.strip()
            f.write("%s\n" % cur_line)
            print(cur_line)
            if cur_line == "":
                continue

            key = cur_line.split()[0]
            if key == previous_key:
                for new_line in new_lines:
                    f.write("%s\n" % new_line)
                    print(new_line)


def replace_line_in_configfile(path, new_line, target_key):
    """
    Parameters
    ----------
    path: str
    line: str
    target_key: str
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
    Parameters
    ----------
    path_in: str
    path_out: str
    exception_names: list[str]
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


def get_hocon_config(config_path, config_name):
    print("Initializing config: {}".format(config_name))
    config = pyhocon.ConfigFactory.parse_file(config_path)[config_name]
    # writelog(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


############################
# General


def get_basename_without_ext(path):
    """
    Parameters
    ----------
    path: str

    Returns
    -------
    str
    """
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]


def get_current_time():
    """
    Returns
    -------
    str
    """
    # return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return datetime.datetime.now().strftime("%b%d_%H-%M-%S")


def get_random_english_word():
    """
    Returns
    -------
    str
    """
    path = os.path.join(os.path.dirname(__file__), "englishwords.txt")
    words = read_lines(path)
    word = np.random.choice(words)
    return word


def hash_string(text):
    """
    Parameters
    ----------
    text: str

    Returns
    -------
    int
    """
    h = hashlib.sha256(text.encode()).hexdigest()
    i = str(int(h, 16))
    return int(i[:8]) # to limit the value between 0 and 2***32-1


class StopWatch(object):

    def __init__(self):
        self.dictionary = {}

    def start(self, name=None):
        """
        Parameters
        ----------
        name: str or None, default None
        """
        start_time = time.time()
        self.dictionary[name] = {}
        self.dictionary[name]["start"] = start_time

    def stop(self, name=None):
        """
        Parameters
        ----------
        name: str or None, default None
        """
        stop_time = time.time()
        self.dictionary[name]["stop"] = stop_time

    def get_time(self, name=None, minute=False):
        """
        Parameters
        ----------
        name: str or None, default None
        minute: bool, default False
        """
        start_time = self.dictionary[name]["start"]
        stop_time = self.dictionary[name]["stop"]
        span = stop_time - start_time
        if minute:
            span /= 60.0
        return span


############################
# IO


def mkdir(path, newdir=None):
    """
    Parameters
    ----------
    path: str
    newdir: str or None, default None
    """
    if newdir is None:
        target = path
    else:
        target = os.path.join(path, newdir)
    if not os.path.exists(target):
        os.makedirs(target)
        writelog("Created directory = %s" % target)


def read_vocab(path):
    """
    Parameters
    ----------
    path: str

    Returns
    -------
    dict[str, int]
    """
    begin_time = time.time()
    writelog("Loading a vocabulary from %s" % path)
    vocab = OrderedDict()
    for line in open(path):
        word, word_id, freq = line.strip().split("\t")
        vocab[word] = int(word_id)
    end_time = time.time()
    writelog("Loaded. %f [sec.]" % (end_time - begin_time))
    writelog("Vocabulary size = %d" % len(vocab))
    return vocab


def write_vocab(path, data):
    """
    Parameters
    ----------
    path: str
    data: list[(str, int)]
    """
    with open(path, "w") as f:
        for word_id, (word, freq) in enumerate(data):
            f.write("%s\t%d\t%d\n" % (word, word_id, freq))


def read_lines(path, process=lambda line: line):
    """
    Parameters
    ----------
    path: str
    process: function: str -> Any, default function: str -> str

    Returns
    -------
    list[Any]
    """
    lines = []
    for line in open(path):
        line = line.strip()
        line = process(line)
        lines.append(line)
    return lines


def write_lines(path, lines, process=lambda line: line):
    """
    Parameters
    ----------
    path: str
    lines: list[Any]
    process: function: Any -> str
    """
    with open(path, "w") as f:
        for line in lines:
            line = process(line)
            f.write("%s\n" % line)


def read_csv(path, delimiter, with_head, with_id, encoding="utf-8"):
    """
    Parameters
    ----------
    path: str
    delimiter: str
    with_head: bool
    with_id: bool
    encoding: str, default "utf-8"

    Returns
    -------
    pandas.DataFrame
    """
    header = 0 if with_head else None
    index_col = 0 if with_id else None
    data = pd.read_csv(path, encoding=encoding, delimiter=delimiter, header=header, index_col=index_col)
    return data


def read_json(path, encoding=None):
    """
    Parameters
    ----------
    path: str
    encoding: str or None, default None

    Returns
    -------
    dict[Any, Any]
    """
    if encoding is None:
        with open(path) as f:
            dct = json.load(f)
    else:
        with io.open(path, "rt", encoding=encoding) as f:
            line = f.read()
            dct = json.loads(line)
    return dct


def write_json(path, dct):
    """
    Parameters
    ----------
    path: str
    dct: dict[Any, Any]
    """
    with open(path, "w") as f:
        json.dump(dct, f, indent=4)


def read_jsonlines(path):
    """
    Parameters
    ----------
    path: str

    Returns
    -------
    list[dict[Any, Any]]
    """
    with jsonlines.open(path) as reader:
        dcts = list(reader)
    return dcts


def read_vectors(path):
    """
    Parameters
    ----------
    path: str

    Returns
    -------
    numpy.ndarray(shape=(N, dim), dtype=float)
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
    Parameters
    ----------
    path: str
    vectors: numpy.ndarray(shape=(N, dim), dtype=float)
    """
    with open(path, "w") as f:
        for i in range(len(vectors)):
            vector = vectors[i]
            vector = [str(x) for x in vector]
            vector = " ".join(vector)
            f.write("%s\n" % vector)


def read_dictionary(path, multivals=False, func_key=lambda x: x, func_val=lambda x: x):
    """
    Parameters
    ----------
    path: str
    multivals: bool, default False
    func_key: function: str -> Any
    func_val: function: str -> Any

    Returns
    -------
    dict[Any, Any] or dict[Any, list[Any]]
    """
    d = {}
    for line in open(path):
        line = line.strip().split()
        if multivals:
            if len(line) == 2:
                key, val = line
                vals = [val]
            else:
                key = line[0]
                vals = line[1:]
            d[func_key(key)] = [func_val(val) for val in vals]
        else:
            if len(line) == 2:
                key, val = line
            else:
                key = line[0]
                val = " ".join(line[1:])
            d[func_key(key)] = func_val(val)
    return d


def read_conll(path, keys):
    """
    Parameters
    ----------
    path: str
    keys: list[str]

    Returns
    -------
    list[list[dict[str, str]]]

    Notes
    -----
    CoNLL-X: ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
    CoNLL-U: ID FORM LEMMA UPOS    XPOS   FEATS HEAD DEPREL DEPS  MISC
    """
    sentences = []

    n_items = len(keys)

    sentence = []
    for line in open(path):
        line = line.strip()
        if line.startswith("#"):
            continue
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
    Parameters
    ----------
    path: str
    sentences: list[list[dict[str, str]]]
    """
    with open(path, "w") as f:
        for sentence in sentences:
            for conll_line in sentence:
                items = [conll_line[key] for key in conll_line.keys()]
                f.write("\t".join(items) + "\n")
            f.write("\n")


def convert_conll_to_linebyline_format(path_conll, keys, ID, FORM, POSTAG, HEAD, DEPREL):
    """
    Parameters
    ----------
    path_conll: str
    keys: list[str]
    ID: str
    FORM: str
    POSTAG: str
    HEAD: str
    DEPREL: str

    Returns
    -------
    list[str]
    list[str]
    list[(int, int, str)]
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


def transform_columnwisedict_to_rowwisedict(dictionary, key_of_keys, key_of_vals, func_key=lambda x: x, func_val=lambda x: x):
    """
    Parameters
    ----------
    dictionary: dict[str, list[str]]
    key_of_keys: str
    key_of_vals: str
    func_key: function: str -> Any
    func_val: function: str -> Any

    Returns
    -------
    dict[Any, Any]

    Examples
    --------
    >> utils.transform_columnwisedict_to_rowwisedict(
            dictionary={"ID": ["0", "1", "2"],
                        "text": ["hello world", "colorless green ideas", "sleep furiously"]},
            key_of_keys="ID",
            key_of_vals="text",
            func_key=lambda x: int(x),
            func_val=lambda x: x.split())
    {0: ["hello", "world"], 1: ["colorless", "green", "ideas"], 2: ["sleep", "furiously"]}
    """
    new_dictionary = {}
    for raw_key, raw_val in zip(dictionary[key_of_keys], dictionary[key_of_vals]):
        # NOTE: raw_key(str), raw_val(str)
        key = func_key(raw_key)
        val = func_val(raw_val)
        new_dictionary[key] = val
    return new_dictionary


def print_list(lst, with_index=False, process=None):
    """
    Parameters
    ----------
    lst: list[Any]
    with_index: bool, default False
    process: function: Any -> Any
    """
    for i, x in enumerate(lst):
        if process is not None:
            x = process(x)
        if with_index:
            print("%d:" % i, x)
        else:
            print(x)


def print_dict(dictionary):
    """
    Parameters
    ----------
    dictionary: dict[Any, Any]
    """
    for key in dictionary.keys():
        print("%s: %s" % (key, dictionary[key]))


def pretty_format_dict(dct):
    """
    Parameters
    ----------
    dct: dict[Any, Any]

    Returns
    -------
    str
    """
    return "{}".format(json.dumps(dct, indent=4))


############################
# Numerical computation


def safe_div(x, y):
    """
    Parameters
    ----------
    x: float or numpy.ndarray
    y: float or numpy.ndarray

    Returns
    -------
    float or numpy.ndarray
    """
    if isinstance(x, np.ndarray):
        mask = y == 0
        x[mask] = 0
        y[mask] = 1
        return x / y
    else:
        if y == 0:
            return 0
        else:
            return x / y


def normalize_vectors(mat):
    """
    Parameters
    ----------
    mat: numpy.ndarray(shape=(batch, feat_dim))

    Returns
    -------
    numpy.ndarray(shape=(batch, feat_dim))
    """
    return mat / np.linalg.norm(mat, axis=1)[:,None]


def levenshtein_distance(seq1, seq2):
    """
    Parameters
    ----------
    seq1: list[Any]
    seq2: list[Any]

    Returns
    -------
    float
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
# Array manipulation


def filter_by_condition(xs, ys, condition_function):
    """
    Parameters
    ----------
    xs: list[list[Any]]
    ys: list[Any]
    condition_function: function: list[Any] -> bool

    Returns
    -------
    list[Any]
    """
    assert len(xs) == len(ys)
    indices = [i for i, x in enumerate(xs) if condition_function(x)]
    zs = [ys[i] for i in indices]
    return zs


def flatten_lists(list_of_lists):
    """
    Parameters
    ----------
    list_of_lists: list[list[Any]]

    Returns
    -------
    list[Any]
    """
    return [elem for lst in list_of_lists for elem in lst]


def get_boundary_indicators_for_sorted_array(sorted_array):
    """
    Parameters
    ----------
    sorted_array: list[int]

    Returns
    -------
    list[bool]
    list[bool]
    """
    start_indicators = [True]
    for i in range(1, len(sorted_array)):
        if sorted_array[i] != sorted_array[i - 1]:
            start_indicators.append(True)
        else:
            start_indicators.append(False)

    end_indicators = [True]
    for i in range(len(sorted_array) - 2, -1, -1):
        if sorted_array[i] != sorted_array[i + 1]:
            end_indicators.append(True)
        else:
            end_indicators.append(False)
    end_indicators = end_indicators[::-1]

    return start_indicators, end_indicators


def compare_dictionary_keys(dict1, dict2):
    """
    Parameters
    ----------
    dict1: dict[Any, Any]
    dict2: dict[Any, Any]

    Returns
    -------
    bool
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    if len(keys1 & keys2) == len(keys1) == len(keys2):
        return True
    else:
        False


def random_replace_list(xs, ps, z):
    """
    Parameters
    ----------
    xs: list[Any]
    ps: float or list[float]
    z: Any

    Returns
    -------
    list[Any]
    """
    N = len(xs)
    if isinstance(ps, float):
        ps = np.zeros((N,)) + ps
    else:
        assert len(ps) == N
    rs = np.random.random((N,))
    ys = [z if r < p else x for x,p,r in zip(xs,ps,rs)]
    return ys


class DataInstance(object):

    def __init__(self, **kargs):
        self.attr_names = []
        for key, value in kargs.items():
            setattr(self, key, value)
            self.attr_names.append(key)

    def __str__(self):
        return "DataInstance(%s)" % ",".join(self.attr_names)


def filter_dataset(dataset, condition):
    """
    Parameters
    ----------
    dataset: numpy.ndarray(shape=(dataset_size,), dtype="O")
    condition: function: DataInstance -> bool

    Returns
    -------
    numpy.ndarray(shape=(dataset_size,), dtype="O")
    """
    filtered_dataset = []
    for data in dataset:
        if condition(data):
            filtered_dataset.append(data)
    filtered_dataset = np.asarray(filtered_dataset, dtype="O")
    return filtered_dataset


def split_dataset(dataset, n_dev, seed=None):
    """
    Parameters
    ----------
    dataset: numpy.ndarray(shape=(dataset_size,), dtype="O")
    n_dev: int
    seed: int or None, default None

    Returns
    -------
    numpy.ndarray(shape=(dataset_size - n_dev,), dtype="O")
    numpy.ndarray(shape=(n_dev,), dtype="O")
    """
    n_total = len(dataset)
    assert 0 < n_dev < n_total

    if seed is None:
        indices = np.random.permutation(n_total)
    else:
        indices = np.random.RandomState(seed).permutation(n_total)

    dev_indices = indices[:n_dev]
    train_indices = indices[n_dev:]

    assert len(train_indices) + len(dev_indices) == len(dataset)

    train_dataset = dataset[train_indices]
    dev_dataset = dataset[dev_indices]

    return train_dataset, dev_dataset


def kfold_dataset(dataset, n_splits, split_id):
    """
    Parameters
    ----------
    dataset: numpy.ndarray(shape=(dataset_size,), dtype="O")
    n_splits: int
    split_id: int

    Returns
    -------
    numpy.ndarray(shape=(train_size,), dtype="O")
    numpy.ndarray(shape=(dev_size,), dtype="O")
    """
    assert 0 <= split_id < n_splits

    kfold = KFold(n_splits=n_splits, random_state=1234, shuffle=True)

    indices_list = list(kfold.split(np.arange(len(dataset))))
    train_indices, dev_indices = indices_list[split_id]
    assert len(train_indices) + len(dev_indices) == len(dataset)

    train_dataset = dataset[train_indices]
    dev_dataset = dataset[dev_indices]

    return train_dataset, dev_dataset


class DataBatch(object):
    """
    Notes
    -----
    Deprecated.
    """

    def __init__(self, **kargs):
        self.attr_names = []
        length = None
        for key, value in kargs.items():
            setattr(self, key, value)
            self.attr_names.append(key)
            # Check
            if length is None:
                length = len(value)
            else:
                assert length == len(value)

    def __len__(self):
        return len(getattr(self, self.attr_names[0]))


def concat_databatch(databatch1, databatch2):
    """
    Parameters
    ----------
    databatch1: DataBatch
    databatch2: DataBatch

    Returns
    -------
    DataBatch

    Notes
    -----
    Deprecated.
    """
    attr_names1 = set(databatch1.attr_names)
    attr_names2 = set(databatch2.attr_names)
    shared_attr_names = attr_names1 & attr_names2
    shared_attr_names = list(shared_attr_names)
    shared_attr_names.sort()
    kargs = {}
    for attr_i, attr_name in enumerate(shared_attr_names):
        writelog("Shared attribute #%d %s" % (attr_i+1, attr_name))
        array1 = getattr(databatch1, attr_name)
        array2 = getattr(databatch2, attr_name)
        new_array = np.concatenate([array1, array2], axis=0)
        assert len(new_array) == len(array1) + len(array2)
        kargs[attr_name] = new_array
    databatch = DataBatch(**kargs)
    return databatch


def filter_databatch(databatch, filtering_function):
    """
    Parameters
    ----------
    databatch: DataBatch
    filtering_function: function: databatch, int -> bool

    Returns
    -------
    DataBatch

    Notes
    -----
    Deprecated.
    """
    kargs = {}
    for attr_name in databatch.attr_names:
        kargs[attr_name] = []

    for entry_i in range(len(databatch)):
        do_filter = filtering_function(databatch, entry_i)
        if not do_filter:
            for attr_name in databatch.attr_names:
                kargs[attr_name].append(getattr(databatch, attr_name)[entry_i])

    for attr_name in databatch.attr_names:
        kargs[attr_name] = np.asarray(kargs[attr_name], dtype="O")

    new_databatch = DataBatch(**kargs)
    return new_databatch


class DataPool(object):
    """
    Notes
    -----
    Deprecated.
    """

    def __init__(self, paths, processes=None, pool_size=1000000):
        """
        Parameters
        ----------
        paths: list[str]
        processes: list[function]
        pool_size: int, default 1000000
        """
        self.paths = paths

        if processes is None:
            self.processes = [lambda l: l for _ in range(self.paths)]
        else:
            assert len(processes) == len(self.paths)
            self.processes = processes

        self._pool_attr_names = ["pool_%d" % path_i for path_i in range(len(self.paths))]

        # Count the number of lines in the text files
        writelog("Counting the number of lines in the text files ...")
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
        Parameters
        ----------
        batch_size: int

        Returns
        -------
        list[numpy.ndarray(shape=(batch_size,), dtype="O")]
        """
        indices = np.random.choice(self.pool_size, size=batch_size) # NOTE that ``replace'' is True.
        output = [getattr(self, pool_attr_name)[indices] for pool_attr_name in self._pool_attr_names]
        self._fill_pools(indices=indices)
        return output

    def _get_init_iterator(self):
        return zip(*[open(path) for path in self.paths])

    def _process(self, tpl):
        """
        Parameters
        ----------
        tpl: tuple[str]

        Returns
        -------
        list[Any]
        """
        return [process(line.strip()) for line, process in zip(tpl, self.processes)]

    def _read_line(self):
        """
        Returns
        -------
        list[Any]
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
        Parameters
        ----------
        indices: list[int] or None, default None
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
        Parameters
        ----------
        n_instances: int

        Returns
        -------
        list[list[Any]]
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
# Machine learning


class TemplateFeatureExtractor(object):

    def __init__(self):
        self.templates = [] # list of str
        self.template2dim = None # {str: int}
        self.feature_size = None # int
        self.UNK_TEMPLATE_DIM = None # int

    ####################################
    def aggregate_templates(self, args):
        """
        Parameters
        ----------
        args: Any

        Returns
        -------
        list[str]
        """
        pass # NOTE: To be defined.

    def add_template(self, **kargs):
        """
        Parameters
        ----------
        kargs: dict[str, str]
        """
        template = self.convert_to_template(**kargs)
        if not template in self.templates:
            self.templates.append(template)

    def convert_to_template(self, **kargs):
        """
        Parameters
        ----------
        kargs: dict[str, str]

        Returns
        -------
        str
        """
        lst = ["%s=%s" % (key,val) for key,val in kargs.items()]
        lst = ",".join(lst)
        template = "<%s>" % lst
        return template
    ####################################

    ####################################
    def prepare(self):
        self.template2dim = {template:dim for dim,template in enumerate(self.templates)}
        self.feature_size = len(self.templates)
        self.UNK_TEMPLATE_DIM = self.feature_size
    ####################################

    ####################################
    def extract_features(self, args):
        """
        Parameters
        ----------
        args: Any

        Returns
        -------
        numpy.ndarray(shape=(1, feature_size), dtype=np.float32)
        """
        # NOTE: To be defined.
        templates = self.generate_templates(args=args)
        template_dims = [self.template2dim.get(t, self.UNK_TEMPLATE_DIM) for t in templates]
        vector = make_multihot_vectors(self.feature_size+1, [template_dims]) # (1, feature_size+1)
        vector = vector[:,:-1] # (1, feature_size)
        return vector

    def extract_batch_features(self, batch_args):
        """
        Parameters
        ----------
        batch_args: list[Any]

        Returns
        -------
        numpy.ndarray(shape=(batch_size, feature_size), dtype=np.float32)
        """
        # NOTE: To be defined.
        fire = [] # list of list of int
        # batch_size = len(batch_args)
        for index, args in enumerate(batch_args):
            templates = self.generate_templates(args=args)
            template_dims = [self.template2dim.get(t, self.UNK_TEMPLATE_DIM) for t in templates]
            fire.append(template_dims)
        vectors = make_multihot_vectors(self.feature_size+1, fire) # (batch_size, feature_size+1)
        vectors = vectors[:,:-1] # (batch_size, feature_size)
        return vectors

    def generate_templates(self, args):
        """
        Parameters
        ----------
        args: Any

        Returns
        -------
        list[str]
        """
        pass # NOTE: To be defined.
    ####################################


def make_multihot_vectors(dim, fire):
    """
    Parameters
    ----------
    dim: int
    fire: list[list[int]]

    Returns
    -------
    numpy.ndarray(shape=(N, dim), dtype=np.float32)
    """
    n_instances = len(fire)
    vectors = np.zeros((n_instances, dim), dtype=np.float32)
    for instance_i in range(n_instances):
        vectors[instance_i, fire[instance_i]] = 1.0
    return vectors


class BestScoreHolder(object):

    def __init__(self, scale=1.0, higher_is_better=True):
        """
        Parameters
        ----------
        scale: float, default 1.0
        higher_is_better: bool, default True
        """
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
        """
        Parameters
        ----------
        score: float
        step: int

        Returns
        -------
        bool
        """
        if self.comparison_function(self.best_score, score):
            # Update the score
            writelog("(best_score = %.02f, best_step = %d, patience = %d) -> (%.02f, %d, %d)" % \
                    (self.best_score * self.scale, self.best_step, self.patience,
                     score * self.scale, step, 0))
            self.best_score = score
            self.best_step = step
            self.patience = 0
            return True
        else:
            # Increment the patience
            writelog("(best_score = %.02f, best_step = %d, patience = %d) -> (%.02f, %d, %d)" % \
                    (self.best_score * self.scale, self.best_step, self.patience,
                     self.best_score * self.scale, self.best_step, self.patience+1))
            self.patience += 1
            return False

    def ask_finishing(self, max_patience):
        """
        Parameters
        ----------
        max_patience: int

        Returns
        -------
        bool
        """
        if self.patience >= max_patience:
            return True
        else:
            return False


############################
# NLP


class BoW(object):

    def __init__(self, documents, tfidf):
        """
        Parameters
        ----------
        documents: list[list[str]]
        tfidf: bool
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
        Parameters
        ----------
        documents: list[list[str]]

        Returns
        -------
        numpy.ndarray(shape=(N,|V|), dtype=np.float32)
        """
        X = self.vectorizer.transform([" ".join(d) for d in documents])
        return X.toarray().astype(np.float32)


def read_english_stopwords():
    """
    Returns
    -------
    list[str]
    """
    stopwords = read_lines(os.path.join(os.path.dirname(__file__), "stopwords.txt"))
    return stopwords


def read_word_embedding_matrix(path, dim, vocab, scale):
    """
    Parameters
    ----------
    path: str
    dim: int
    vocab: dict[str, int]
    scale: float

    Returns
    -------
    numpy.ndarray(shape=(vocab_size, dim), dtype=np.float32)
    """
    word2vec = read_word2vec(path, dim)
    W = convert_word2vec_to_weight_matrix(vocab, word2vec, dim, scale)
    return W


def read_word2vec(path, dim=None):
    """
    Parameters
    ----------
    path: str
    dim: int or None, default None

    Returns
    -------
    dict[str, numpy.ndarray(shape=(dim,), dtype=np.float32)]
    """
    writelog("Loading pretrained word vectors from %s ..." % path)

    word2vec = {}

    # Determine the dimension size if dim is None
    if dim is None:
        for line_i, line in enumerate(open(path)):
            # Check the second line to ignore the top head line
            if line_i == 1:
                items = line.strip().split()
                dim = len(items[1:])
                break
    writelog("Dimensionality = %d" % dim)

    # Prepara prog_bar
    n_lines = 0
    for _ in open(path):
        n_lines += 1
    prog_bar = pyprind.ProgBar(n_lines)

    # Make a dictionary from word to vector
    error_history = []
    with open(path) as f:
        for line_i, line in enumerate(f):
            items = line.strip().split()
            if len(items[1:]) != dim:
                error_history.append(
                                {"dim_actual": len(items[1:]),
                                 "line_id": line_i+1,
                                 "line": ",".join(items[:10])}
                                )
                # print("dim %d(actual) != %d(expected), skipped %d-th line=%s..." % \
                #         (len(items[1:]), dim, line_i+1, ",".join(items[:10])))
                continue
            word2vec[items[0]] = np.asarray([float(x) for x in items[1:]])
            prog_bar.update()

    for err in error_history:
        writelog("dim %d(actual) != %d(expected), skipped %d-th line = %s ....." % \
                    (err["dim_actual"], dim, err["line_id"], err["line"]))

    writelog("Vocabulary size = %d" % len(word2vec))
    writelog("%s" % prog_bar)
    return word2vec


def convert_word2vec_to_weight_matrix(vocab, word2vec, dim, scale):
    """
    Parameters
    ----------
    vocab: dict[str, int]
    word2vec: dict[str, numpy.ndarray(shape=(dim,), dtype=np.float32)]
    dim: int
    scale: float

    Returns
    -------
    numpy.ndarray(shape=(vocab_size, dim), dtype=np.float32)
    """
    writelog("Converting ...")
    begin_time = time.time()

    task_vocab = list(vocab.keys())
    word2vec_vocab = list(word2vec.keys())
    shared_vocab = set(task_vocab) & set(word2vec_vocab)
    writelog("Vocabulary size (task) = %d" % len(task_vocab))
    writelog("Vocabulary size (word2vec) = %d" % len(word2vec_vocab))
    writelog("Vocabulary size (shared) = %d (|shared|/|task| = %d/%d = %.2f%%)" % \
            (len(shared_vocab), len(shared_vocab), len(task_vocab),
                float(len(shared_vocab))/len(task_vocab)*100.0))

    # NOTE: If we fix the word vectors, we should use the same random seed for initializing the out-of-vocabulary words.
    W = np.random.RandomState(1234).uniform(-scale, scale, (len(task_vocab), dim)).astype(np.float32)
    for w in shared_vocab:
        W[vocab[w], :] = word2vec[w]

    end_time = time.time()
    writelog("Converted. %f [sec.]" % (end_time - begin_time))
    return W


def read_word2vec_using_gensim(path, binary):
    """
    Parameters
    ----------
    path: str
    binary: bool

    Returns
    -------
    gensim.models.keyedvectors.Word2VecKeyedVectors
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)
    return model


def keyedvectors2dict(model):
    """
    Parameters
    ----------
    model: gensim.models.keyedvectors.Word2VecKeyedVectors

    Returns
    -------
    dict[str, numpy.ndarray(shape=(dim,), dtype=np.float32)]
    """
    word2vec = {}
    vocab = model.vocab
    for word in vocab.keys():
        word2vec[word] = model[word]
    return word2vec


def read_process_and_write(paths_in, paths_out, process: lambda line: line):
    """
    Parameters
    ----------
    paths_in: list[str]
    paths_out: list[str]
    process: function: str -> str
    """
    assert len(paths_in) == len(paths_out)
    n_files = len(paths_in)
    prog_bar = pyprind.ProgBar(n_files)
    for path_in, path_out in zip(paths_in, paths_out):
        with open(path_out, "w") as f:
            for line in open(path_in):
                line = line.strip()
                line = process(line)
                f.write("%s\n" % line)
        prog_bar.update()


def build_vocabulary(paths_file, path_vocab, prune_at, min_count, special_words, process=lambda line: line.strip().split(), unk_symbol=None, with_unk=True):
    """
    Parameters
    ----------
    paths_file: list[str]
    path_vocab: str
    prune_at: int
    min_count: int
    special_words: list[str]
    process: function: str -> list[str]
    unk_symbol: str or None, default None
    with_unk: bool, default True
    """
    assert not os.path.exists(path_vocab)

    if unk_symbol is None:
        unk_symbol = "<unk>"

    # Count
    counter = Counter()
    for path_file in pyprind.prog_bar(paths_file):
        for line in open(path_file):
            tokens = process(line)
            counter.update(tokens)
    counter = counter.most_common()

    # Prune
    counter = counter[:prune_at]
    frequencies = dict(counter)
    counter.sort(key=lambda x: (-x[1], x[0]))
    vocab_words = [w for w,freq in counter if freq >= min_count]

    # Add special words
    for sw in special_words:
        if not sw in vocab_words:
            vocab_words = vocab_words + [sw]
            frequencies[sw] = 0 # TODO

    # Creat a word-to-id dictionary
    vocab = OrderedDict()
    for w_id, w in enumerate(vocab_words):
        vocab[w] = w_id

    # Add a special OOV symbol
    if with_unk and not unk_symbol in vocab.keys():
        vocab[unk_symbol] = len(vocab)
        frequencies[unk_symbol] = 0 # TODO

    if with_unk:
        writelog("Vocabulary size (w/ '%s') = %d" % (unk_symbol, len(vocab)))
    else:
        writelog("Vocabulary size = %d" % len(vocab))

    # Write
    with open(path_vocab, "w") as f:
        for w, w_id in vocab.items():
            freq = frequencies[w]
            f.write("%s\t%d\t%d\n" % (w, w_id, freq))

    writelog("Saved the vocabulary to %s" % path_vocab)


def concat_vocabularies(paths_vocab, path_out):
    """
    Parameters
    ----------
    paths_vocab: list[str]
    path_out: str
    """
    assert len(paths_vocab) > 1

    vocab = OrderedDict()

    for path_vocab in paths_vocab:
        for line in open(path_vocab):
            word, word_id, freq = line.strip().split("\t")
            freq = int(freq)
            if word in vocab:
                word_id_exst, freq_exst = vocab[word]
                freq_exst = int(freq_exst)
                vocab[word] = (word_id_exst, freq_exst + freq)
            else:
                vocab[word] = (word_id, freq)

    with open(path_out, "w") as f:
        for word in vocab.keys():
            word_id, freq = vocab[word]
            f.write("%s\t%s\t%d\n" % (word, word_id, freq))

    writelog("Saved the vocabulary to %s" % path_out)


def replace_oov_tokens(paths_in, paths_out, path_vocab, unk_symbol=None):
    """
    Parameters
    ----------
    paths_in: list[str]
    paths_out: list[str]
    path_vocab: str
    unk_symbol: str or None, default None
    """
    assert len(paths_in) == len(paths_out)

    if unk_symbol is None:
        unk_symbol = "<unk>"

    vocab = read_vocab(path_vocab)
    vocab = list(vocab.keys())

    vocab = {w:w for w in vocab}

    prog_bar = pyprind.ProgBar(len(paths_in))
    for path_in, path_out in zip(paths_in, paths_out):
        lines = read_lines(path_in, process=lambda line: line.split())

        lines = [[vocab.get(token, unk_symbol) for token in line] for line in lines]

        with open(path_out, "w") as f:
            for line in lines:
                line = " ".join(line)
                f.write("%s\n" % line)

        prog_bar.update()


def get_word_counter(path=None, lines=None):
    """
    Parameters
    ----------
    path: str or None, default None
    lines: list[list[str]] or None, default None

    Returns
    -------
    collections.Counter
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
    Parameters
    ----------
    path_dir: str
    top_k: int
    process: function: str -> list[str]
    """
    filenames = os.listdir(path_dir)

    # stopwords = read_lines(os.path.join(os.path.dirname(__file__), "stopwords.txt"))
    stopwords = read_english_stopwords()

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
        writelog("word = %s, frequency = %d" % (w, freq))


def normalize_string(string, able=None):
    """
    Parameters
    ----------
    string: str
    able: list[str] or None, default None

    Returns
    -------
    str
    """
    if able is None:
        return string
    if "space" in able:
        string = re.sub(r"[\t\u2028\u2029\u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000]+", " ", string)
    if "hyphen" in able:
        string = re.sub(r"[\u2010\u002d\u2011\u2043\u2212]+", "-", string)
    if "amp" in able:
        string = re.sub(r"&amp;", "&", string)
    if "quot" in able:
        string = re.sub(r"&quot;", "'", string)
    if "lt" in able:
        string = re.sub(r"&lt;", "<", string)
    if "gt" in able:
        string = re.sub(r"&gt;", ">", string)
    return string


############################
# Chainer

# Deprecated!!!

def transform_words(xs):
    """
    Parameters
    ----------
    xs: list[list[int]]

    Returns
    -------
    list[Variable(shape=(L,), dtype=np.int32)]

    Notes
    -----
    Deprecated.
    """
    xs = [np.asarray(x, dtype=np.int32) for x in xs]
    xs = [Variable(cuda.cupy.asarray(x)) for x in xs]
    return xs


def padding(xs, head, with_mask):
    """
    Parameters
    ----------
    xs: list[list[int]]
    head: bool
    with_mask: bool

    Returns
    -------
    numpy.ndarray(shape=(N, max_length))
    numpy.ndarray(shape(N, max_length))
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
    Parameters
    ----------
    xs: numpy.ndarray(shape=(N, L))
    seq: bool

    Returns
    -------
    list[Variable(shape=(N,))] or Variable(shape=(N, L))

    Notes
    -----
    Deprecated.
    """
    if seq:
        return [Variable(cuda.cupy.asarray(xs[:,j]))
                for j in range(xs.shape[1])]
    else:
        return Variable(cuda.cupy.asarray(xs))


# def get_optimizer(name):
#     """
#     :type name: str
#     :rtype: chainer.Optimizer
#     """
#     if name == "sgd":
#         opt = optimizers.SGD()
#     elif name == "momentumsgd":
#         opt = optimizers.CorrectedMomentumSGD()
#     elif name == "adadelta":
#         opt = optimizers.AdaDelta()
#     elif name == "adagrad":
#         opt = optimizers.AdaGrad()
#     elif name == "adam":
#         opt = optimizers.Adam()
#     elif name == "rmsprop":
#         opt = optimizers.RMSprop()
#     elif name == "rmspropgraves":
#         opt = optimizers.RMSpropGraves()
#     elif name == "smorms3":
#         opt = optimizers.SMORMS3()
#     else:
#         raise ValueError("Unknown optimizer_name=%s" % name)
#     return opt


############################
# Analysis


def extract_values_with_regex(filepath, regex, names):
    """
    Parameters
    ----------
    filepath: str
    regex: str
    names: list[str]

    Returns
    -------
    dict[str, list[str]]
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


# def calc_score_stats(filepaths, regex, names):
#     """
#     :type filepaths: list of str
#     :type regex: str
#     :type names: list of str
#     :rtype: Pandas.DataFrame
#     """
#     columns_raw = {name: [] for name in names} # {str: list of float}
#     for filepath in filepaths:
#         scores = extract_values_with_regex(filepath, regex, names) # {str: list of str}
#         for name in names:
#             assert len(scores[name]) == 1
#             score = float(scores[name][0])
#             columns_raw[name].append(score)
#     # e.g., {precision: [file1_p, file2_p, file3_p],
#     #        recall: [file1_r, file2_r, file3_r]}
#
#     columns = OrderedDict()
#     columns["Method"] = [os.path.basename(filepath) for filepath in filepaths] + ["mean", "std"]
#     for name in names:
#         columns[name] = columns_raw[name] + [np.mean(columns_raw[name]), np.std(columns_raw[name])]
#     # e.g., {Method: [file1, file2, file3, "mean", "std"],
#     #        precision: [file1_p, file2_p, file3_p, mean_p, std_p],
#     #        recall: [file1_r, file2_r, file3_r, mean_r, std_r],
#
#     for name1 in names:
#         max_index = np.argmax(columns_raw[name1])
#         for name2 in names:
#             columns[name2].append(columns_raw[name2][max_index])
#         columns["Method"].append("Max-%s: %s" % (name1, os.path.basename(filepaths[max_index])))
#     # e.g., {Method: [file1, file2, file3, "mean", "std", "Max-precision: file*", "Max-recall: file**"],
#     #        precision: [file1_p, file2_p, file3_p, mean_p, std_p, file*_p, file**_p],
#     #        recall: [file1_r, file2_r, file3_r, mean_r, std_r, file*_r, file**_r],
#
#     df = pd.DataFrame(columns)
#     pd.options.display.float_format = "{:,.2f}".format
#     return df


# def plot_given_files(
#         filepaths, regex,
#         xticks, xlabel, ylabels,
#         legend_names, legend_anchor, legend_location,
#         marker="o", linestyle="-", markersize=10,
#         fontsize=30,
#         savepaths=None, figsize=(8,6), dpi=100):
#     """
#     :type filepaths: list of str
#     :type regex: str
#     :type xticks: list of str
#     :type xlabel: str
#     :type ylabels: list of str
#     :type legend_names: list of str
#     :type legend_anchor: (int, int)
#     :type legend_location: str
#     :type marker: str
#     :type linestyle: str
#     :type markersize: int
#     :type fontsize: int
#     :type savepaths: list of str
#     :type figsize: (int, int)
#     :type dpi: int
#     :rtype: None
#     """
#     assert len(filepaths) == len(legend_names)
#
#     # Extraction
#     data = {ylabel: [] for ylabel in ylabels} # {str: list of list of float}
#     for filepath in filepaths:
#         scores = extract_values_with_regex(filepath, regex, ylabels) # {str: list of str}
#         for ylabel in ylabels:
#             data[ylabel].append([float(x) for x in scores[ylabel]])
#
#     if savepaths is None:
#         savepaths = [None for _ in range(len(ylabels))]
#
#     for ylabel, savepath in zip(ylabels, savepaths):
#         visualizers.plot(
#                     list_ys=data[ylabel], list_xs=None,
#                     xticks=xticks, xlabel=xlabel, ylabel=ylabel,
#                     legend_names=legend_names,
#                     legend_anchor=legend_anchor, legend_location=legend_location,
#                     marker=marker, linestyle=linestyle, markersize=markersize,
#                     fontsize=fontsize,
#                     savepath=savepath, figsize=figsize, dpi=dpi)
#

