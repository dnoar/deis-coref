import pandas as pd
from nltk.tree import Tree
from nltk.corpus import stopwords
from proinfo import SING,PLUR,PRONOUN_LIST,NUMBER_DICT,ANIMATE_DICT,ANIMATE,NONANIMATE,PERSON_DICT

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
        return ('','')
            
    if targetTree.label() != 'NP':
        return ('','')
        
    bestHead = ''
    pos = None
    while targetTree != '':
        looking = 1
        nextSubtree = ''
        for child in targetTree:
            if type(child) == Tree and child.label() == 'NP' and looking:
                nextSubtree = child
                looking = 0
            if type(child) == Tree and child.label() in ('NN','NNP','NNS'):
                bestHead = child.leaves()[0]
                pos = child.label()
        if bestHead != '':
            return (bestHead,pos)
        targetTree = nextSubtree
        
    return (bestHead,pos)

def find_mention_head(df,mention,tree):
    t = Tree.fromstring(tree)
    
    sent_num,word_span = mention
    
    target = build_word_span(df,sent_num,word_span)
    if len(target.split(' ')) == 1:
        return (target,get_features(df,sent_num,int(word_span))[5])
        
    head,pos = find_head(t,target)
    
    return (head.lower(),pos)
    
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
            
        head = find_mention_head(df,firstMention,trees[(filename,str(part_num),str(firstMention[0]))])[0]
        if head == '':
            newGroupingsDict[i] = groupings[i]
            continue
        
        for key,cluster in newGroupingsDict.items():
            foundGoodHead = 0
            antContent = set()
            for antecedent in cluster:
                if head == find_mention_head(df,antecedent,trees[(filename,str(part_num),str(antecedent[0]))])[0]:
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
    
def build_indices(groupings):
    """Build indexed, ordered lists of mentions partitioned by sentence, as well as a second partitioned by grouping"""
    
    sentenceDict = {}
    index = {}
    
    for i in range(len(groupings)):
        grouping = groupings[i]
        for sent_num,word_span in grouping:
            if sent_num in sentenceDict:
                sentenceDict[sent_num].append((sent_num,word_span))
            else:
                sentenceDict[sent_num] = [(sent_num,word_span)]
            index[(sent_num,word_span)] = i
            
    for sent in sentenceDict:
        sentenceDict[sent].sort(key=(lambda x: int(x[1].split('_')[0])))
        
    return (sentenceDict,index)
    
def find_number(mention,df,sent_tree):
    
    numberSet = set()
    
    #first check the NER tag
    ner = get_features(df,mention[0],int(mention[1].split('_')[0]))[11].strip('()*')
    if ner in ('LOC','PERSON','GPE','FAC'):
        numberSet.add(SING)
        return numberSet
    elif ner == 'ORG':
        numberSet.add(SING)
        numberSet.add(PLUR)
        return numberSet
    
    #then check POS of the head
    pos = find_mention_head(df,mention,sent_tree)[1]
    if pos == 'NNS':
        numberSet.add(PLUR)
    else:
        numberSet.add(SING)
    
            
    return numberSet
            
def find_animacy(mention,df):

    ner = get_features(df,mention[0],int(mention[1].split('_')[0]))[11].strip('()*')
    if ner == 'PERSON':
        return ANIMATE
    else:
        return NONANIMATE
    
def check_property_match(pronoun,mention,df,sentTree):

    
    #Check if number matches
    mentionNum = find_number(mention,df,sentTree)
    if len(mentionNum) > 0 and len(set(NUMBER_DICT[pronoun]).intersection(mentionNum)) == 0:
        return False
    
    #Check if animacy matches    
    if find_animacy(mention,df) not in ANIMATE_DICT[pronoun]:
        return False
    
    #just those two, at least for now    
    #person is only for pronoun-pronoun, gender needs world knowledge
    
    return True
    
def check_property_match_pro(a,b,oneInQuotes):

    #only scenario where the same pronoun would refer to a different person, so we put it before the string-match
    if (not oneInQuotes) and PERSON_DICT[a] != PERSON_DICT[b]:
        return False
    
    #if they're the same they're the same, what are you gonna do
    if a == b:
        return True
        
    #Check number agreement
    if len(set(NUMBER_DICT[a]).intersection(set(NUMBER_DICT[b]))) == 0:
        return False
    
    #Check animacy agreement
    if len(set(ANIMATE_DICT[a]).intersection(set(ANIMATE_DICT[b]))) == 0:
        return False
        
    #add gender
    if len(set(GENDER_DICT[a]).intersection(set(GENDER_DICT[b]))) == 0:
        return False
        
    return True
    
def is_in_quotes(df,sent_num,word_span):
    
    firstWord = int(word_span.split('_')[0])
    sentSpan = ''
    for i in range(int):
        sentSpan += str(i) + '_'
    sentSpan = sentSpan.strip('_')
    if sentSpan == 0:
        return False
    sentPart = build_word_span(df,sent_num,sentSpan)
    quoteCount = sentPart.count('"')
    if quoteCount % 2 == 0:
        return False
    return True
    
def module7(groupings,df,trees,filename,part_num):
    
    sentences,index = build_indices(groupings)
    
    for i in range(len(groupings)):
        #unMerged = 1
        sent_num,word_span = groupings[i][0]
        pronoun = build_word_span(df,sent_num,word_span)
        #now we're ONLY interested in pronouns
        if pronoun not in PRONOUN_LIST:
            continue
        
        #we only want to check this sentence (R->L) and the previous (L->R)
        currSent = sentences[sent_num]
        currSent = currSent[:currSent.index((sent_num,word_span))]
        currSent.reverse()
        inQuotes = is_in_quotes(df,sent_num,word_span)
        
        currSentTree = trees[filename,str(part_num),str(sent_num)]
        matched = 0
        for mention in currSent:
            
            
            if text in PRONOUN_LIST:
                textInQuotes = is_in_quotes(df,mention[0],mention[1])
                check_property_match_pro(pronoun,text,inQuotes or textInQuotes)
                
            if not check_property_match(pronoun,mention,df,currSentTree):
                continue
            
            matched = 1
            rightCluster = index[mention]
            for proclust in groupings[i]:
                index[proclust] = rightCluster
            break
        
        #done with this cluster if we matched it, or this is the first sentence
        if matched or sent_num == 0 or (sent_num - 1) not in sentences:
            continue
            
        #otherwise we're looking at the previous sentence (in its original order), IF it has any mentions! >:0
        prevSent = sentences[sent_num-1]
        prevSentTree = trees[filename,str(part_num),str(sent_num-1)]
        
        for mention in prevSent:
            text = build_word_span(df,mention[0],mention[1])
            if text in PRONOUN_LIST:
                textInQuotes = is_in_quotes(df,mention[0],mention[1])
                check_property_match_pro(pronoun,text,inQuotes or textInQuotes)
                
            if not check_property_match(pronoun,mention,df,prevSentTree):
                continue
                
            rightCluster = index[mention]
            for proclust in groupings[i]:
                index[proclust] = rightCluster
            break
    
    #everything should be situated at this point; just rebuild the groupings from the index
            
    newGroupings = []        
    for i in range(max(index.values())+1):
        newGroupings.append([])
        
    for mention in index:
        newGroupings[index[mention]].append(mention)
    
    while [] in newGroupings:
        newGroupings.remove([])
        
    return newGroupings
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
#etc    