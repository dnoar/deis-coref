import pandas as pd
from nltk.tree import Tree
from nltk.corpus import stopwords
from proinfo import SING,PLUR,PRONOUN_LIST,NUMBER_DICT,ANIMATE_DICT,ANIMATE,NONANIMATE,PERSON_DICT,GENDER_DICT,M,F,N
from npfeats import quick_check_logs

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

def build_word_span(df,sent_num,word_span,lowerize=True):
    """Given the identifiers, get the real words that make up this mention in a single string separated by spaces. Default is to lowercase it, but sometimes that's no good."""
    wordList = []
    for word_num in word_span.split('_'):
        wordList.append(get_features(df,sent_num,int(word_num))[4])
    
    if lowerize:
        return ' '.join(wordList).lower()
    else:
        return ' '.join(wordList)
    
def find_sisters(t,target):
    """Find the syntactic sisters of the given phrase in the given tree.
    Goes through the subtrees of the tree, adding its children to a list. If it finds a subtree whose leaves match the target,
    we assume that is the constituent of our target. Then we can return the list of sisters. If it doesn't find the subtree
    on one level, it will clear the sister-list and try again."""
    
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
    """Find any mentions that were appositives of the given mention (but only before the mention, not after)"""
    
    sent_num,word_span = mention
    
    #First word in our original mention
    begin = int(word_span.split('_')[0])
    
    #no prior appositives if this is the 1st or 2nd word in the sentence
    if begin < 2:
        return None
    
    #Build the target from the words for the mention, and find its sisters in the syntactic tree
    target = build_word_span(df,sent_num,word_span)
    sisters = find_sisters(Tree.fromstring(tree),target)
    
    #not the cases we're looking for
    if len(sisters) < 1 or sisters[0] != "NP":
        return None
        
    #Only accept an appositive if it's the first NP in a list of sisters of [NP,",",NP] with  no conjunction in the list
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
    if df.loc[(df['sent_num'] == sent_num) & (df['word_num'] == (begin-1-len(appos.split()))),'corefs'].get_values()[0] == '-': #only want appositives if they're actually mentions
        return None
    elif df.loc[(df['sent_num'] == sent_num) & (df['word_num'] == (begin-2)),'corefs'].get_values()[0] == '-': #and the entire span needs to be a mention
        return None
        
    #Build the span to match it to the mention format and send it packing
    for i in range(begin-1-len(appos.split()),begin-1):
        appos_span += str(i)+'_'
    return (sent_num,appos_span.strip('_'))
    
def acro_info(df,mention):
    """Returns the acronym form of this mention (if it can be turned into an acronym)"""
    sent_num,word_span = mention
    
    #Only find the acronym for non-acronym mentions if they're all proper nouns
    for word_num in word_span.split('_'):
        pos = get_features(df,sent_num,int(word_num))[5]
        if pos != 'NNP':
            return ''
        
    start = build_word_span(df,sent_num,word_span,lowerize=False)
    
    #Acronyms have to be all-uppercase and greater than 2 characters
    if start.isupper() and len(start) > 2:
        return start
    
    #Acronymables have to be in title case and greater than 2 words
    elif start.istitle() and len(start.split()) > 2:
        acro = ''
        for word in start.split():
            acro += word[0]
        return acro
    else:
        return ''
        
def find_head(t,target):
    """Find the head of the target phrase in the given sentence tree.
    First, find the appropriate NP tree for the target(if possible).
    Then, find the highest NN, NNP, or NNS, and if there are multiple at the same level,
    take the furthest-right one. Search the NPs breadth-first, left-to-right.
    We also want to return the POS in order to determine if the target is singular or plural
    when pronoun-matching."""
    
    
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
    """Find the head word and its POS of the given mention using the syntax tree"""
    t = Tree.fromstring(tree)
    
    sent_num,word_span = mention
    
    target = build_word_span(df,sent_num,word_span)
    
    #With just one word, it's its own head
    if len(target.split(' ')) == 1:
        return (target,get_features(df,sent_num,int(word_span))[5])
        
    head,pos = find_head(t,target)
    
    return (head.lower(),pos)
    
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
    """Find the number (singular or plural or both) for a mention."""
    
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
    """Find the animacy (animate or nonanimate) for a mention."""

    ner = get_features(df,mention[0],int(mention[1].split('_')[0]))[11].strip('()*')
    if ner == 'PERSON':
        return ANIMATE
    else:
        return NONANIMATE
    
def check_property_match(pronoun,mention,df,sentTree,g_log):
    """Check if the properties of the pronoun we're trying to group match those of the mention we're trying to group it with."""
    
    #Check if number matches
    mentionNum = find_number(mention,df,sentTree)
    if len(mentionNum) > 0 and len(set(NUMBER_DICT[pronoun]).intersection(mentionNum)) == 0:
        return False
    
    #Check if animacy matches    
    if find_animacy(mention,df) not in ANIMATE_DICT[pronoun]:
        return False
    
    #person is only for pronoun-pronoun
    '''
    #Check if gender matches - unused, did not improve
    gender = quick_check_logs(build_word_span(df,mention[0],mention[1]),g_log)[0]
    if gender == 'female' and F not in GENDER_DICT[pronoun]:
        return False
    elif gender == 'male' and M not in GENDER_DICT[pronoun]:
        return False
    elif gender == 'neutral' and N not in GENDER_DICT[pronoun]:
        return False
    '''
    
    return True
    
def check_property_match_pro(a,b,oneInQuotes):
    """Check whether two pronouns match properties."""

    #Check person; however, if one is in quotes, then we don't check person because it is most likely different speakers, so different-person pronouns can actually match
    #This is the only scenario where two pronouns that don't match in person could still match, or the same pronoun could be for different speakers,
    #so we put it before the string match.
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
        
    #Check gender agreement
    if len(set(GENDER_DICT[a]).intersection(set(GENDER_DICT[b]))) == 0:
        return False
        
    return True
    
def is_in_quotes(df,sent_num,word_span):
    """Check if a word span is in quotes by counting the number of quotes before it in the sentence. If an odd number, that means an unclosed quote."""
    
    firstWord = int(word_span.split('_')[0])
    
    #Nothing before the first word in a sentence
    if firstWord == 0:
        return False
        
    #Build the sentence prior to the given span and count its quotes
    sentSpan = ''
    for i in range(firstWord):
        sentSpan += str(i) + '_'
    sentSpan = sentSpan.strip('_')
    sentPart = build_word_span(df,sent_num,sentSpan)
    quoteCount = sentPart.count('"')
    
    #Even number of qoutes = none open when our word_span is hit
    if quoteCount % 2 == 0:
        return False
        
    return True

def module1(groupings,df,trees,filename,part_num):
    """First module: Exact match of words (except pronouns)"""
    matchDict = {}
    newGroupings = []
    proCount = 0
    
    for grouping in groupings:
        
        mention = grouping[0]
        words = build_word_span(df,mention[0],mention[1])
        
        #do NOT want to deal with pronouns this pass
        if words in PRONOUN_LIST: 
            matchDict["pro" + str(proCount)] = [mention]
            proCount += 1
        
        else:
            if words in matchDict:
                matchDict[words].append(mention)
            else:
                matchDict[words] = [mention]
    
    #For each grouping, sort by sentence number primarily, and word numbers secondarily
    for key in matchDict:
        mentionList = matchDict[key]
        mentionList.sort(key=(lambda x: int(x[1].split('_')[0])))
        mentionList.sort(key=(lambda x: x[0]))
        newGroupings.append(mentionList)
    
    
    #sort groupings of mentions by text appearance order
    newGroupings.sort(key=(lambda x: int(x[0][1].split('_')[0])))
    newGroupings.sort(key=(lambda x: x[0][0]))
    
    return newGroupings
    
    

def module2(groupings,df,trees,filename,part_num):
    """Second module: Appositives and acronyms"""
    
    newGroupingsDict = {}
    newGroupings = []
    
    
    for i in range(len(groupings)):
        unMerged = 1
        mention = groupings[i][0]
        
        #Find appositives, if any
        appositive = find_prior_appositives(df,mention,trees[(filename,str(part_num),str(mention[0]))])
        
        #Find the acronym form, if any
        acronymForm = acro_info(df,mention)
        
        #Search through the mentions already seen and grouped (should be earlier in the sentence too, since we sorted them at the end of module 1)
        for key,cluster in newGroupingsDict.items():
            
            #Check each mention in each cluster individually. If it matches the appositive to our current mention, or matches as acronym, then merge them up.
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
                    
        #Either add this grouping to an existing grouping, or start a new one if it hasn't found a match
        if unMerged:
            newGroupingsDict[i] = groupings[i]
        else:
            newGroupingsDict[rightCluster].extend(groupings[i])
            
    #Rewrite all the new groupings and return
    keyList = list(newGroupingsDict.keys())
    keyList.sort()
    for key in keyList:
        newGroupings.append(newGroupingsDict[key])
    
    return newGroupings
    
def module3(groupings,df,trees,filename,part_num):
    """Third module: Head matching"""
    
    newGroupingsDict = {}
    newGroupings = []
    stopwordsList = stopwords.words('english')
    
    for i in range(len(groupings)):
        unMerged = 1
        
        #We only want to match up the first (earliest-in-text) mention for each group
        firstMention = groupings[i][0]
        
        #Ignore pronouns
        if build_word_span(df,firstMention[0],firstMention[1]) in PRONOUN_LIST:
            newGroupingsDict[i] = groupings[i]
            continue
        
        #Find all the non-stop-words in this grouping's mentions
        contentList = set()
        for mention in groupings[i]:
            words = build_word_span(df,mention[0],mention[1])
            for word in words.split(' '):
                if word not in stopwordsList:
                    contentList.add(word)
        
        #If it doesn't have anything but stopwords it's not going to be helpful
        if len(contentList) == 0:
            newGroupingsDict[i] = groupings[i]
            continue
        
        #Find this mention's headword, if possible
        head = find_mention_head(df,firstMention,trees[(filename,str(part_num),str(firstMention[0]))])[0]
        if head == '':
            newGroupingsDict[i] = groupings[i]
            continue
        
        #For each cluster, add all of its non-stop-words to a set, and try to find a headword that matches the headword for our current mention.
        #If both conditions are satisfied, merge them together.
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
    
    #Rewrite all the new groupings and return
    keyList = list(newGroupingsDict.keys())
    keyList.sort()
    for key in keyList:
        newGroupings.append(newGroupingsDict[key])
    
    return newGroupings
    
def module7(groupings,df,trees,filename,part_num,g_log):
    """Fourth module: Pronoun matching"""
    
    #Need indices because we're now going by strict sentence order, instead of sentence order within grouping order
    sentences,index = build_indices(groupings)
    
    for i in range(len(groupings)):
        
        sent_num,word_span = groupings[i][0]
        pronoun = build_word_span(df,sent_num,word_span)
        
        #now we're ONLY interested in pronouns
        if pronoun not in PRONOUN_LIST:
            continue
        
        #we only want to check this sentence (R->L) and the previous (L->R)
        currSent = sentences[sent_num]
        currSent = currSent[:currSent.index((sent_num,word_span))]
        currSent.reverse()
        
        #See if the pronoun is in quotes (can matter when matching pronoun-pronoun)
        inQuotes = is_in_quotes(df,sent_num,word_span)
        
        currSentTree = trees[filename,str(part_num),str(sent_num)]
        matched = 0
        
        #Compare pronoun to all the mentions in the sentence before it
        for mention in currSent:
            
            text = build_word_span(df,mention[0],mention[1])
            
            #Pronoun-pronoun
            if text in PRONOUN_LIST:
                textInQuotes = is_in_quotes(df,mention[0],mention[1])
                if not check_property_match_pro(pronoun,text,inQuotes or textInQuotes):
                    continue
            
            #Pronoun-noun
            elif not check_property_match(pronoun,mention,df,currSentTree,g_log):
                continue
            
            #Match these up
            matched = 1
            rightCluster = index[mention]
            for proclust in groupings[i]:
                index[proclust] = rightCluster
            break
        
        #done with this cluster if we matched it, or this is the first sentence
        if matched or sent_num == 0 or (sent_num - 1) not in sentences:
            continue
            
        #otherwise we're looking at the previous sentence (in its original order), IF it has any mentions
        prevSent = sentences[sent_num-1]
        prevSentTree = trees[filename,str(part_num),str(sent_num-1)]
        
        #Use same method as current sentence
        for mention in prevSent:
            text = build_word_span(df,mention[0],mention[1])
            if text in PRONOUN_LIST:
                textInQuotes = is_in_quotes(df,mention[0],mention[1])
                if not check_property_match_pro(pronoun,text,inQuotes or textInQuotes):
                    continue
                
            elif not check_property_match(pronoun,mention,df,prevSentTree,g_log):
                continue
                
            rightCluster = index[mention]
            for proclust in groupings[i]:
                index[proclust] = rightCluster
            break
    
    #everything should be situated at this point; just rebuild the groupings from the index
    
    #Make an index in the list for each new list
    newGroupings = []        
    for i in range(max(index.values())+1):
        newGroupings.append([])
    
    #Add each mention to the appropriate grouping
    for mention in index:
        newGroupings[index[mention]].append(mention)
    
    #Collapse the list of lists
    while [] in newGroupings:
        newGroupings.remove([])
        
    return newGroupings