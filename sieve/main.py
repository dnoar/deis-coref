import pickle
import pandas as pd
import sieve_modules
import csv
from string import punctuation
from preprocess import get_trees

IMPLEMENTED_MODULES = [sieve_modules.module1,sieve_modules.module2,sieve_modules.module3]

def print_groupings(f,part_df,filename,part_num,groupings):
    
    f.write("#begin document (" + filename + "); part " + format(part_num,'03') + '\n')
    
    for i,chain in enumerate(groupings):
        for sent_num,word_span in chain:
            sent_num = int(sent_num)
            #multi-word mentions
            if '_' in str(word_span):
                start = int(word_span.split('_')[0])
                end = int(word_span.split('_')[-1])
                
                corefStart = part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == start),'corefs'].get_values()[0]
                corefEnd = part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == end),'corefs'].get_values()[0]
                
                if corefStart == '-':
                    print(filename,part_num,sent_num,word_span)
                if corefEnd == '-':
                    print(filename,part_num,sent_num,word_span)
                
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
                if coref == '-':
                    print(filename,part_num,sent_num,word_span)
                if coref.startswith('_'):
                    part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == word_span),'corefs'] = coref + '|('+str(i)+')'
                else:
                    part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == word_span),'corefs'] = '_('+str(i)+')'
    
    sentNum = 0
    
    for array in part_df.get_values():
        line = ''
        for i,value in enumerate(array):
            if i == 2:
                if value > sentNum:
                    f.write('\n')
                    sentNum = value
                continue
            if i == 12: #this is where we're printing the corefs
                line += '*\t*\t*\t'
                value = value.strip('_')
            line += str(value) + '\t'
            
        f.write(line.strip() + '\n')
        
    f.write("\n#end document\n")
            
def merge_groupings(groupings):
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

if __name__ == '__main__':
    with open('dev.pickle','rb') as f:
        coref_chains = pickle.load(f)
    
    df = pd.read_csv('./coref_dev.feat')
    trees = get_trees('./coref_dev.feat')
    
    for filename in coref_chains:
        file_chains = coref_chains[filename]
        
        with open('./new_results/' + filename.split('/')[-1]+'.results','w',encoding='utf8') as f:
            for part_num in file_chains:
                #mentions = []
                groupings = []
                part_chains = file_chains[part_num]
                
                sub_df = df[(df['doc_id'] == filename) & (df['part_num'] == part_num)]
                
                #get all the mentions for this filename/part
                #to start with, every mention is in its own grouping
                for chain in part_chains:
                    chain_groupings = [[(tuple[0],str(tuple[1]))] for tuple in part_chains[chain]]
                    groupings.extend(chain_groupings)
                
                if groupings != []:
                    for module in IMPLEMENTED_MODULES:
                        groupings = module(groupings,sub_df,trees,filename,part_num)
                        #if module != sieve_modules.module1:
                         #   groupings = merge_groupings(groupings)
            
                print_groupings(f,sub_df,filename,part_num,groupings)
                
        
        