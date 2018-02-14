#Extract entity pairs

from __future__ import print_function
import os
import csv
import random
from nltk.tree import Tree
from string import punctuation
import pandas as pd

SAMPLE_ANNOTATION = '../conll-2012/train/english/annotations/bc/cctv/00/cctv_0001.v4_auto_conll'

"""Note that _conll files have the follwowing format:
Column	Type	Description
1	Document ID	This is a variation on the document filename
2	Part number	Some files are divided into multiple parts numbered as 000, 001, 002, ... etc.
3	Word number
4	Word itself	This is the token as segmented/tokenized in the Treebank. Initially the *_skel file contain the placeholder [WORD] which gets replaced by the actual token from the Treebank which is part of the OntoNotes release.
5	Part-of-Speech
6	Parse bit	This is the bracketed structure broken before the first open parenthesis in the parse, and the word/part-of-speech leaf replaced with a *. The full parse can be created by substituting the asterix with the "([pos] [word])" string (or leaf) and concatenating the items in the rows of that column.
7	Predicate lemma	The predicate lemma is mentioned for the rows for which we have semantic role information. All other rows are marked with a "-"
8	Predicate Frameset ID	This is the PropBank frameset ID of the predicate in Column 7.
9	Word sense	This is the word sense of the word in Column 3.
10	Speaker/Author	This is the speaker or author name where available. Mostly in Broadcast Conversation and Web Log data.
11	Named Entities	These columns identifies the spans representing various named entities.
12:N	Predicate Arguments	There is one column each of predicate argument structure information for the predicate mentioned in Column 7.
N	Coreference	Coreference chain information encoded in a parenthesis structure.
"""

FEATURE_NAMES = ("doc_id", "part_num","sent_num",
            "word_num", "word", "pos",
            "parse_bit", "pred_lemma", "pred_frame_id",
            "sense","speaker","ne","corefs"
            )

def featurize_file(filename):
    with open(filename) as source:
        """Convert gold_conll file to features
        Input:
            filename: path to gold_conll file
        Output:
            featurized_words: a list of dicts of features, one dict per word in the file
        TODO: feature processing (mostly strings now)
        """
        featurized_words = list()
        sent_count = 0

        for line in source:
            attribs = line.split()  #See comments at beginning of file for conll column format
            #print(attribs) #for debugging
            if len(attribs) == 0: #If it's a blank line, we're starting a new sentence
                sent_count += 1
                #print("On sent {}".format(sent_count)) #for debugging
            elif not attribs[0].startswith('#'): #if it's not a comment
                feature_dict = dict()
                sent_num = sent_count
                doc_id, part_num, word_num, word, pos = attribs[:5]
                parse_bit, pred_lemma, pred_frame_id, sense, speaker, ne = attribs[5:11]
                args = attribs[11:-1] #a list
                corefs = attribs[-1] #list of entity numbers,parens,and pipes e.g. (28), (42, 64), (28|(42
                features = (doc_id, part_num, sent_num,word_num,word,pos,parse_bit,pred_lemma,pred_frame_id,sense,speaker,ne,corefs)
                featurized_sent = {k:v for k,v in zip(FEATURE_NAMES,features)}
                featurized_words.append(featurized_sent)
        return featurized_words


def featurize_dir(dirname):
    """Featurize all files in a dir
    Input:
        dirname: path to conll directory
    Output:
        list of list of dicts
    """
    featurized_files = list()
    for root, dirnames, filenames in os.walk(dirname):
        for filename in filenames:
            if filename.endswith('gold_conll'):
                featurized_files.append(featurize_file(os.path.join(root,filename)))
    return featurized_files

def write_csv(featurized_files):
    """Write featurized files to csv
    """
    with open('coref.feat','w',newline='') as dest:
        writer = csv.DictWriter(dest, fieldnames=FEATURE_NAMES)
        writer.writeheader()
        for featurized_file in featurized_files:
            for featurized_word in featurized_file:
                row = dict()
                for feature_name in FEATURE_NAMES:
                    feature = featurized_word[feature_name]
                    if type(feature) is list:
                        feature = ''.join(feature)
                    row[feature_name] = feature
                writer.writerow(row)

def get_trees(featfile):
    """Get trees from a feature file
    """
    trees = dict()
    with open(featfile) as source:
        reader = csv.DictReader(source)
        for row in reader:
            key = (row['doc_id'],row['part_num'],row['sent_num'])
            word = row['word']
            if word in punctuation:
                word = '(PUNC {})'.format(word)
            try:
                trees[key] += row['parse_bit'].replace('*',' ' + word)
            except KeyError:
                trees[key] = row['parse_bit'].replace('*',' ' + word)
    return trees

def get_nps(featfile):
    """Get trees from a feature file, then get NPs from trees
    Use these NPs to build coreference chains
    """
    np_dict = dict()
    trees = get_trees(featfile)
    for key in trees.keys():
        tree = Tree.fromstring(trees[key])
        nps = list(tree.subtrees(filter=lambda x:x.label() == "NP"))
        for np in nps:
            np_string = '_'.join(np.leaves())
            try:
                np_dict[key].append(np_string)
            except KeyError:
                np_dict[key] = [np_string]
    return np_dict

def build_coref_chains(featfile):
    """Build coreference chains from featurized files
    """
    corefs = dict() #keys: (file, num). values: (sent, word_span)
    df = pd.read_csv(featfile)
    filenames = df.doc_id.unique()
    for filename in filenames:
        file_df = df[df[doc_id] == filename]
        corefs = file_dif.corefs.unique()
        sents = file_df.sent_num.unique()
    #TODO: not finished yet

if __name__ == "__main__":
    """
    print("Featurizing...")
    featurized_files = featurize_dir('../conll-2012/test/')
    print("Writing csv...")
    write_csv(featurized_files)
    """

    """
    print("Extracting coreference chains...")
    coref_dicts = build_coref_chains(featurized_files)
    print(coref_dicts[100].keys())
    for key in coref_dicts[100].keys():
        print(coref_dicts[key])
    """

    print("Getting NPs")
    nps = get_nps('../train.feat')
    sample = nps[('nw/wsj/02/wsj_0290', '0', '47')]
    print(sample)
