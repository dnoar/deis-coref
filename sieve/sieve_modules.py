import pandas as pd

PRONOUN_LIST = ['i','me','my','mine','you','your','yours','he','him','his','she','her','hers','we','us','our','ours','they','them','their','theirs','it','its']

def get_features(df,doc_id,part_num,sent_num,word_num):
    """Get the list of features for this specific row in the data"""
    return df[(df['doc_id'] == doc_id) &
              (df['part_num'] == part_num) &
              (df['sent_num'] == sent_num) &
              (df['word_num'] == word_num)].get_values()[0]
              
def get_features_row(df,doc_id,part_num,sent_num,word_num):
    """Get the list of features for this specific row in the data"""
    return df[(df['doc_id'] == doc_id) &
              (df['part_num'] == part_num) &
              (df['sent_num'] == sent_num) &
              (df['word_num'] == word_num)]

def build_word_span(df,filename,part_num,sent_num,word_span):
    """given the identifiers, get the words that make up this mention"""
    wordList = []
    for word_num in str(word_span).split('_'):
        wordList.append(get_features(df,filename,part_num,sent_num,int(word_num))[4].lower())
    
    return ' '.join(wordList)

def module1(groupings,df,filename,part_num):
    """First module: Exact match of words"""
    matchDict = {}
    newGroupings = []
    proCount = 0
    
    for grouping in groupings:
        tuple = grouping[0]
        words = build_word_span(df,filename,part_num,tuple[0],tuple[1])
        if words in PRONOUN_LIST: #do NOT want to deal with pronouns this pass
            matchDict["pro" + str(proCount)] = [tuple]
            proCount += 1
        else:
            if words in matchDict:
                matchDict[words].append(tuple)
            else:
                matchDict[words] = [tuple]
            
    for key in matchDict:
        newGroupings.append(matchDict[key])
    
    return newGroupings
    
def module2(groupings,df,filename,part_num):
    """Second module: Appositives, predicate nominative, acronym"""
    matchDict = {}
    newGroupings = []
    proCount = 0
    
    for grouping in groupings:
        
    
def module3(args):
    pass
    
#etc    