import pickle
import pandas as pd
import sieve_modules

IMPLEMENTED_MODULES = [sieve_modules.module1]

def print_groupings(f,df,filename,part_num,groupings):
    
    part_df = df[(df['doc_id'] == filename) & (df['part_num'] == part_num)]
    f.write("#begin document (" + filename + "); part " + format(part_num,'03') + '\n')
    
    for i,chain in enumerate(groupings):
        for sent_num,word_span in chain:
        
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
                coref = part_df.loc[(part_df['sent_num'] == sent_num) & (part_df['word_num'] == word_span),'corefs'].get_values()[0]
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
            

if __name__ == '__main__':
    with open('dev.pickle','rb') as f:
        coref_chains = pickle.load(f)
    
    df = pd.read_csv('./coref_dev.feat')
    
    
    
    for filename in coref_chains:
        file_chains = coref_chains[filename]
        
        with open('./new_results/' + filename.split('/')[-1]+'.results','w',encoding='utf8') as f:
            for part_num in file_chains:
                #mentions = []
                groupings = []
                part_chains = file_chains[part_num]
                
                #get all the mentions for this filename/part
                #to start with, every mention is in its own grouping
                for chain in part_chains:
                    chain_groupings = [[tuple] for tuple in part_chains[chain]]
                    groupings.extend(chain_groupings)
                    
                for module in IMPLEMENTED_MODULES:
                    groupings = module(groupings,df,filename,part_num)
            
                print_groupings(f,df,filename,part_num,groupings)
                
        