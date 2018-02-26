import pandas as pd
from nltk.tree import Tree
from nltk.corpus import stopwords

PRONOUN_LIST = ['i','me','my','mine','you','your','yours','he','him','his','she','her','hers','we','us','our','ours','they','them','their','theirs','it','its','this','that','those','these']

def get_features(df,sent_num,word_num):
    """Get the list of features for this specific row in the data
    Returns a list of features:
        0 = doc_id
        1 = part_num
        2 = sent_num
        3 = word_num
        4 = word
        5 = pos
        6 = parse_bit
        7 = pred_lemma
        8 = pred_frame_id
        9 = sense
        10 = speaker
        11 = ne
        12 = corefs
    """
    return df[(df['sent_num'] == sent_num) &
              (df['word_num'] == word_num)].get_values()[0]
              
def get_features_row(df,sent_num,word_num):
    """Get the list of features for this specific row in the data"""
    return df[(df['sent_num'] == sent_num) &
              (df['word_num'] == word_num)]

def build_word_span(df,sent_num,word_span,lowerize=True):
    """given the identifiers, get the words that make up this mention"""
    wordList = []
    for word_num in word_span.split('_'):
        wordList.append(get_features(df,sent_num,int(word_num))[4])
    
    if lowerize:
        return ' '.join(wordList).lower()
    else:
        return ' '.join(wordList)
    
def find_sisters(t,target):

    sisters = []
    targetFound = 0
    for subtree in t.subtrees():
        sisters.append(subtree.label())
        for child in subtree:
            if type(child) == Tree:
                leaves = ' '.join(child.leaves()).lower()
            else:
                leaves = child.lower()
            if leaves == target:
                targetFound = 1
            sisters.append(leaves)
        if targetFound:
            break
        else:
            sisters = []
    
    return sisters
    
def find_prior_appositives(df,mention,tree):
    sent_num,word_span = mention
    
    begin = int(word_span.split('_')[0])
    end = int(word_span.split('_')[-1])
    
    #no prior appositives if this is the 1st or 2nd word in the sentence
    if begin < 2:
        return None
    
    #beginFeatures = get_features(df,filename,part_num,sent_num,begin)
    #endFeatures = get_features(df,filename,part_num,sent_num,end)
    
    target = build_word_span(df,sent_num,word_span)
    sisters = find_sisters(Tree.fromstring(tree),target)
    
    #not the cases we're looking for
    if len(sisters) < 1 or sisters[0] != "NP":
        return None
        
    conjFound = 0
    priorComma = 0
    preTarget = 1
    appos = ''
    for sister in sisters[1:]:
        if sister == target:
            if (appos == '' or not priorComma):
                appos = ''
                break
            else:
                preTarget = 0
        elif sister == ',':
            priorComma = 1
        elif sister in ('and','but','or'):
            appos = ''
            break
        elif appos == '' and preTarget:
            appos = sister # restricing to one appositive
            
    if appos == '':
        return None
    
    appos_span = ''
    if df.loc[(df['sent_num'] == sent_num) & (df['word_num'] == (begin-1-len(appos.split()))),'corefs'].get_values()[0] == '-':
        return None
    elif df.loc[(df['sent_num'] == sent_num) & (df['word_num'] == (begin-2)),'corefs'].get_values()[0] == '-':
        return None
        
    for i in range(begin-1-len(appos.split()),begin-1):
        appos_span += str(i)+'_'
    return (sent_num,appos_span.strip('_'))
    


def module1(groupings,df,trees,filename,part_num):
    """First module: Exact match of words"""
    matchDict = {}
    newGroupings = []
    proCount = 0
    
    for grouping in groupings:
        tuple = grouping[0]
        words = build_word_span(df,tuple[0],tuple[1])
        if words in PRONOUN_LIST: #do NOT want to deal with pronouns this pass
            matchDict["pro" + str(proCount)] = [tuple]
            proCount += 1
        else:
            if words in matchDict:
                matchDict[words].append(tuple)
            else:
                matchDict[words] = [tuple]
            
    for key in matchDict:
        mentionList = matchDict[key]
        mentionList.sort(key=(lambda x: int(x[1].split('_')[0])))
        mentionList.sort(key=(lambda x: x[0]))
        newGroupings.append(mentionList)
    
    
    #sort mentions by text appearance order
    newGroupings.sort(key=(lambda x: int(x[0][1].split('_')[0])))
    newGroupings.sort(key=(lambda x: x[0][0]))
    
    return newGroupings
    
    
def acro_info(df,mention):
    sent_num,word_span = mention
    
    for word_num in word_span.split('_'):
        pos = get_features(df,sent_num,int(word_num))[5]
        if pos != 'NNP':
            return ''
        
    start = build_word_span(df,sent_num,word_span,lowerize=False)
    
    if start.isupper() and len(start) > 2:
        return start
    elif start.istitle() and len(start.split()) > 2:
        acro = ''
        for word in start.split():
            acro += word[0]
        return acro
    else:
        return ''
        
def find_head(t,target):
    """oversimplicated head-finding"""
    
    
    
    targetFound = 0
    for subtree in t.subtrees():
        for child in subtree:
            if type(child) == Tree:
                leaves = ' '.join(child.leaves()).lower()
            else:
                leaves = child.lower()
            if leaves == target:
                targetFound = 1
                targetTree = child
        if targetFound:
            break
    
    if targetFound == 0:
        return ''
            
    if targetTree.label() != 'NP':
        return ''
        
    bestHead = ''
    while targetTree != '':
        looking = 1
        nextSubtree = ''
        for child in targetTree:
            if type(child) == Tree and child.label() == 'NP' and looking:
                nextSubtree = child
                looking = 0
            if type(child) == Tree and child.label() in ('NN','NNP','NNS'):
                bestHead = child.leaves()[0]
        if bestHead != '':
            return bestHead
        targetTree = nextSubtree
        
    return bestHead

def find_mention_head(df,mention,tree):
    t = Tree.fromstring(tree)
    
    sent_num,word_span = mention
    
    target = build_word_span(df,sent_num,word_span)
    if len(target.split(' ')) == 1:
        return target
        
    head = find_head(t,target)
    
    return head.lower()
    
def module2(groupings,df,trees,filename,part_num):
    """Second module: Appositives, predicate nominative, acronym"""
    #matchDict = {}
    newGroupingsDict = {}
    newGroupings = []
    #proCount = 0
    #acronyms = {}
    
    for i in range(len(groupings)):
        unMerged = 1
        mention = groupings[i][0]
        appositive = find_prior_appositives(df,mention,trees[(filename,str(part_num),str(mention[0]))])
        acronymForm = acro_info(df,mention)
        for key,cluster in newGroupingsDict.items():
            for antecedent in cluster:
                if appositive == antecedent:
                    unMerged = 0
                    rightCluster = key
                    break
                if acronymForm != '' and acronymForm == acro_info(df,antecedent) and ('_' not in mention[1] or '_' not in antecedent[1]):
                    unMerged = 0
                    rightCluster = key
                    break
            if not unMerged:
                break
                    
        if unMerged:
            newGroupingsDict[i] = groupings[i]
        else:
            newGroupingsDict[rightCluster].extend(groupings[i])
    keyList = list(newGroupingsDict.keys())
    keyList.sort()
    for key in keyList:
        newGroupings.append(newGroupingsDict[key])
    
    return newGroupings
    
def module3(groupings,df,trees,filename,part_num):
    newGroupingsDict = {}
    newGroupings = []
    stopwordsList = stopwords.words('english')
    
    for i in range(len(groupings)):
        unMerged = 1
        firstMention = groupings[i][0]
        if build_word_span(df,firstMention[0],firstMention[1]) in PRONOUN_LIST:
            newGroupingsDict[i] = groupings[i]
            continue
        
        contentList = set()
        for mention in groupings[i]:
            words = build_word_span(df,mention[0],mention[1])
            for word in words.split(' '):
                if word not in stopwordsList:
                    contentList.add(word)
                    
        if len(contentList) == 0:
            newGroupingsDict[i] = groupings[i]
            continue
            
        head = find_mention_head(df,firstMention,trees[(filename,str(part_num),str(firstMention[0]))])
        if head == '':
            newGroupingsDict[i] = groupings[i]
            continue
        
        for key,cluster in newGroupingsDict.items():
            foundGoodHead = 0
            antContent = set()
            for antecedent in cluster:
                if head == find_mention_head(df,antecedent,trees[(filename,str(part_num),str(antecedent[0]))]):
                    foundGoodHead = 1
                words = build_word_span(df,antecedent[0],antecedent[1])
                for word in words.split(' '):
                    if word not in stopwordsList:
                        antContent.add(word)
            if foundGoodHead and contentList.issubset(antContent):
                unMerged = 0
                rightCluster = key
                break
        if unMerged:
            newGroupingsDict[i] = groupings[i]
        else:
            newGroupingsDict[rightCluster].extend(groupings[i])
    
    keyList = list(newGroupingsDict.keys())
    keyList.sort()
    for key in keyList:
        newGroupings.append(newGroupingsDict[key])
    
    return newGroupings
    
#etc    