#Extract entity pairs

from __future__ import print_function
import pickle
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
    with open(filename,'r',encoding='utf8') as source:
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
    with open('coref.feat','w',encoding='utf8',newline='') as dest:
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
            pos = row['pos']
            if word in punctuation:
                word = '(PUNC {})'.format(word)
            try:
                trees[key] += row['parse_bit'].replace('*',' (' + pos + ' ' + word + ')')
            except KeyError:
                trees[key] = row['parse_bit'].replace('*',' (' + pos + ' ' + word + ')')
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
    Input:
        featfile - Path to file where previous featurization was saved as a CSV
    Returns:
        A three-tiered dictionary, indexed by (in descending order) filename, part number, and coreference chain.
        The bottom tier holds the most pertinent information: Sentence number and word number for each mention in the chain.
        If it is a single-word mention, the word number will be a single value.
        If it is a multi-word mention, the word number will be multiple values spaced by underscore (e.g., 5_6_7).
    """
    #corefs = dict() #keys: (file, num). values: (sent, word_span)
    df = pd.read_csv(featfile)
    filenames = df.doc_id.unique()
    fileDict = dict()
    for filename in filenames:

        file_df = df[df['doc_id'] == filename]
        partnums = file_df.part_num.unique()
        partDict = dict()

        for partnum in partnums:

            chainDict = dict()
            part_df = file_df[file_df['part_num'] == partnum]
            corefs = part_df[part_df['corefs'] != '-']
            for coref in corefs.get_values():
                sentNum = coref[2]
                wordNum = coref[3]
                refNum = coref[-1]

                chainDict = match_corefs(chainDict,refNum,sentNum,wordNum)

            partDict[partnum] = chainDict

        fileDict[filename] = partDict

        #corefs = file_df.corefs.unique()
        #sents = file_df.sent_num.unique()
    return fileDict

def match_corefs(chainDict,newCorefList,sentNum,wordNum):
    """
    Matches an explicit coreference mention to the appropriate chain for this file and part number.
    Input:
        chainDict - A dictionary for each chain, indexed by the coreference chain's number. The values of the dictionary are lists of mentions.
        newCorefList - The direct coref value from the CONLL format (e.g. (124) or (124|(113) or 113|124)
        sentNum - The sentence number of this mention, which is saved as part of the mention info
        wordNum - The word number of this mention, which may be combined with previous word numbers for multi-word mentions
    Returns:
        chainDict, but with the appropriate mentions added to coreference chain(s)
    """

    #One word may be relevant to numerous chains, all split by |
    for newCoref in newCorefList.split('|'):

        #Starting a new mention
        if newCoref.startswith('('):

            #Single-word mention
            if newCoref.endswith(')'):
                refNum = newCoref[1:-1]
                chainDict = add_new_coref(chainDict,refNum,(sentNum,wordNum))

            #Multi-word mention
            else:
                refNum = newCoref[1:]
                if refNum in chainDict:
                    #here we are just saving the wordNum of the current word, which will be the beginning of the multi-word mention span
                    chainDict[refNum].append(wordNum)
                else:
                    chainDict[refNum] = [wordNum]

        #ending a multi-word mention, which will have the format "##)"
        else:
            refNum = newCoref[:-1]

            #get the latest still-open mention, and its number and all of the words in between
            wordBegin = chainDict[refNum].pop()
            span = ''
            for i in range(wordBegin,wordNum+1):
                span += str(i) + '_'
            span = span.strip('_')


            chainDict = add_new_coref(chainDict,refNum,(sentNum,span))

    return chainDict

def add_new_coref(chainDict,refNum,coref):
    """Add a completed single- or multi-word mention to its coreference chain, taking into account
    the fact that there might be still-open multi-word mentions in this same chain
    Input:
        chainDict - A dictionary for each chain, indexed by the coreference chain's number. The values of the dictionary are lists of mentions.
        refNum - The index for chainDict
        coref - The new (sentenceNumber,wordSpan) tuple to add to the coreference chain
    Returns:
        chainDict with coref added in between the previous completed mention (if any) and word numbers for still-open mentions (if any)
    """

    #simplest case, first time we've seen this number and it's a one-word mention
    if refNum not in chainDict:
        chainDict[refNum] = [coref]
        return chainDict

    chainList = chainDict[refNum]
    openList = []

    #Pop out and store all still-open mentions
    if len(chainList) > 0:
        while type(chainList[-1]) == int:
            openList.append(chainList.pop())
            if len(chainList) < 1:
                break

    chainList.append(coref)

    #Add the popped-out still-open mentions back to the chain in their original order
    while len(openList) > 0:
        chainList.append(openList.pop())

    return chainDict

if __name__ == "__main__":

    print("Featurizing...")
    featurized_files = featurize_dir('../conll-2012/train/')
    print("Writing csv...")
    write_csv(featurized_files)



    print("Extracting coreference chains...")
    #coref_dicts = build_coref_chains(featurized_files)
    coref_dicts = build_coref_chains('./coref.FEAT')
    with open('corefs.pickle','wb') as f:
        pickle.dump(coref_dicts,f,pickle.HIGHEST_PROTOCOL)
    #print(coref_dicts[100].keys())
    #for key in coref_dicts[100].keys():
     #   print(coref_dicts[key])

    """
    print("Getting NPs")
    nps = get_nps('../train.feat')
    sample = nps[('nw/wsj/02/wsj_0290', '0', '47')]
    print(sample)
    """
