import pickle
import pandas as pd
import sieve_modules
import csv
from string import punctuation
from preprocess import get_trees

FEAT_PICKLE = '../coref.pickle'
FEAT_FILE = '../coref.feat'
IMPLEMENTED_MODULES = [sieve_modules.module1,sieve_modules.module2,sieve_modules.module3,sieve_modules.module7]

def print_groupings(f,part_df,filename,part_num,groupings):
    """Change the DF of this filename and part number to our found mention chain. Then, print the entire DF.
    Inputs:
      f - File object for the document we're writing to
      part_df - The dataframe filtered to the appropriate filename and part number
      filename - The document name
      part_num - The part number
      groupings - Our list of lists, each element being a list of mentions (a tuple of a sentence number and a word span)
    """
    #Document header
    f.write("#begin document (" + filename + "); part " + format(part_num,'03') + '\n')

    #Change the data frame to reflect our custom groupings
    #The first time we add a grouping number to a particular word, we also add an underscore to the beginning.
    #That way, if we need to add another grouping number to it, we know that it's already been changed to reflect OUR groupings
    for i,chain in enumerate(groupings):
        for sent_num,word_span in chain:
            sent_num = int(sent_num)

            #multi-word mentions
            if '_' in str(word_span):
                start = int(word_span.split('_')[0])
                end = int(word_span.split('_')[-1])

                corefStart = part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == start),'corefs'].get_values()[0]
                corefEnd = part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == end),'corefs'].get_values()[0]


                if corefStart.startswith('_'):
                    part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == start),'corefs'] = corefStart + '|('+str(i)
                else:
                    part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == start),'corefs'] = '_('+str(i)

                if corefEnd.startswith('_'):
                    part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == end),'corefs'] = corefEnd + '|'+str(i)+')'
                else:
                    part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == end),'corefs'] = '_'+str(i)+')'

            else: #single-word mentions
                word_span = int(word_span)
                coref = part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == word_span),'corefs'].get_values()[0]
                if coref.startswith('_'):
                    part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == word_span),'corefs'] = coref + '|('+str(i)+')'
                else:
                    part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == word_span),'corefs'] = '_('+str(i)+')'

    sentNum = 0

    #Write everything in the array to the file
    for array in part_df.get_values():
        line = ''
        for i,value in enumerate(array):

            #This is the sentence number. We don't want to write it because we put it in ourselves, it's not in the CONLL format.
            #However, if it changes, we do want to put in a newline to separate sentences.
            if i == 2:
                if value > sentNum:
                    f.write('\n')
                    sentNum = value
                continue
            #this is where we're printing the corefs and adding a couple of other columns that we don't use
            if i == 12:
                line += '*\t*\t*\t'
                value = value.strip('_')
            line += str(value) + '\t'

        f.write(line.strip() + '\n')

    f.write("\n#end document\n")

'''
def merge_groupings(groupings):
    """DEPRECATED - used to merge sets when groupings could be in multiple sets at once"""
    index = {}
    for i,chain in enumerate(groupings):
        dup_list = []
        for mention in chain:
            if mention in index: #found a duplicate, need to merge sets
                dup_list.append(index[mention])
        if dup_list == []:
            for mention in chain:
                index[mention] = i
        else:
            real_i = min(dup_list)
            for dup_group in dup_list:
                if dup_group == real_i:
                    continue
                for other_mention in groupings[dup_group]:
                    index[other_mention] = real_i
            for mention in chain:
                index[mention] = real_i


    mergedGroupings = []
    for i in range(max(index.values())+1):
        mergedGroupings.append([])

    for mention in index:
        mergedGroupings[index[mention]].append(mention)

    while [] in mergedGroupings:
        mergedGroupings.remove([])

    return mergedGroupings
'''

if __name__ == '__main__':
    #Load the starting coreference chains to get the mentions
    with open('../coref_test.pickle','rb') as f:
        coref_chains = pickle.load(f)

    #Read the features and syntax trees
    df = pd.read_csv(FEATFILE)
    trees = get_trees(FEATFILE)

    #Gender log for pronoun-linking, used in sieve_modules.module7
    #Allows us to look up entity in wikipedia only once
    g_log = {}

    #Go document name by document name, part number by part number
    for filename in coref_chains:
        file_chains = coref_chains[filename]


        with open('../new_results/' + filename.split('/')[-1]+'.results','w',encoding='utf8') as f:
            for part_num in file_chains:

                groupings = []
                part_chains = file_chains[part_num]

                #Don't need to pass the whole dataframe around, since we're only looking at one document/part number pair at a time
                sub_df = df[(df['doc_id'] == filename) & (df['part_num'] == part_num)]

                #get all the mentions for this filename/part
                #to start with, every mention is in its own grouping
                for chain in part_chains:
                    chain_groupings = [[(tuple[0],str(tuple[1]))] for tuple in part_chains[chain]]
                    groupings.extend(chain_groupings)

                #Some parts don't have any mentions and that's ok
                if groupings != []:

                    #For each module, pass in the current groupings and get back the new (and hopefully improved) groupings
                    for module in IMPLEMENTED_MODULES:

                        #Special call for module7, since it needs g_log, which holds data even between documents and part numbers
                        if module == sieve_modules.module7:
                            groupings = module(groupings,sub_df,trees,filename,part_num,g_log)

                        else:
                            groupings = module(groupings,sub_df,trees,filename,part_num)

                #Print our findings
                print_groupings(f,sub_df,filename,part_num,groupings)
