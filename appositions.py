"""
Tool for finding coreferent appositions
"""

import preprocess
from nltk.tree import Tree
import pickle

#Should genitive pronouns be omitted because they could still be part?
PRONOUNS = ['I','my','me','you','your','he','his','she','her','it','its','we','our','they','their','that']

def get_appositions(featfile,corefs):
    """Get a list of appositions from the feature file
    Input:
        featfile: path to the feature file
        corefs: dict of the form dict[filename][part][coref_num]
    Output:
        list of appositions as strings
    """
    appositions = list()
    trees = preprocess.get_trees(featfile) #keys are file, part, sent
    for filename in corefs.keys():
        for part in corefs[filename].keys():
            #list of coref numbers appearing in that part
            coref_nums = list(corefs[filename][part].keys())

            #iterate over the coreference numbers
            for coref_num in coref_nums:
                #part_corefs is list of (sent_num, word_indices) tuples
                #sent is int, word_indices may be int or str
                part_corefs = corefs[filename][part][coref_num]

                #Sentences in which there are words coreferring to the given coref_num
                sent_nums = sorted(list(set([sent_num for sent_num,num_chain in part_corefs])))
                for sent_num in sent_nums:
                    tokens = Tree.fromstring(trees[(filename, str(part), str(sent_num))]).leaves()
                    #print("File {} Part {} Sent {}".format(filename, part, sent_num))

                    #sent_corefs is list of (sent_num, word_indices) tuples
                    #sent is int, word_indices may be int or str
                    sent_corefs = [part_coref for part_coref in part_corefs if part_coref[0] == sent_num]

                    #Sort sent_corefs by index of first word index
                    sorted_corefs = sorted(sent_corefs, key=lambda x: int(str(x[1]).split('_')[0]))
                    if len(sorted_corefs) > 1:
                        #can only have apposition if there's more than one NP in the sentence correfering to the given coref number
                        for i in range(len(sent_corefs)-1):
                            np0_indices = [int(idx) for idx in str(sorted_corefs[i][1]).split('_')]
                            np1_indices = [int(idx) for idx in str(sorted_corefs[i+1][1]).split('_')]
                            if np1_indices[0] - np0_indices[-1] == 2:
                                #If there's just room for a comma between np0 and np1
                                np0 = ' '.join([tokens[np_index] for np_index in np0_indices])
                                np1 = ' '.join([tokens[np_index] for np_index in np1_indices])
                                apposition_candidate = "{} , {}".format(np0,np1)
                                if apposition_candidate in ' '.join(tokens) and not (np0.lower() == np1.lower()) and np0[0].isupper() and not np1.lower() in PRONOUNS:
                                    appositions.append(apposition_candidate)
                                    #print("Found apposition {}".format(apposition_candidate))
    return appositions

if __name__ == '__main__':
    with open('./corefs.pickle','rb') as source:
        corefs = pickle.load(source)
    appositions = get_appositions('../train.feat',corefs)
    with open('appositions.txt','w') as dest:
        for apposition in appositions:
            dest.write("{}\n".format(apposition))
