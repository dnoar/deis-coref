#Extract entity pairs

from __future__ import print_function

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

def extract_entities(filename):
    """Extract entities to get CONLL format to match Latte slide format
    Input:
        filename: path to _conll file
    Output:
        list of (string, coref_number) tuples (TODO: add more/different features)
    TODO: add sentence number in addition to word number,
        since we'll look at sentence/word number when deciding which pairs to look at.
    """
    with open(filename) as source:

        """entities is currently a list of strings separated by _
        TODO: add more features than just the string,
        e.g. the pos tags for each word, tree positions, etc.
        """
        entities = list()

        #The sentence we're on in the file
        sent_count = 0

        #Keys: coref number Values: current string for the entity, separated by _
        entity_strings = dict() #(number, string) dict

        #Keys: coref number Values: whether we're adding words we encounter to the string for that entity
        entity_status = dict() #(num, collecting?) dict

        for line in source:
            attribs = line.split()  #See comments at beginning of file for conll column format
            #print(attribs) #for debugging
            if len(attribs) == 0: #If it's a blank line, we're starting a new sentence
                sent_count += 1
                #print("On sent {}".format(sent_count)) #for debugging
            elif not attribs[0].startswith('#'): #if it's not a comment
                sent_num = sent_count
                doc_id, part_num, word_num, word, pos = attribs[:5]
                parse_bit, pred_lemma, pred_frame_id, sense, speaker, ne = attribs[5:11]
                args = attribs[11:-1] #a list
                corefs = attribs[-1] #list of entity numbers,parens,and pipes e.g. (28), (42, 64), (28|(42
                #print(corefs)

                #If we're starting or ending one or more entities
                if corefs != '-':
                    #List of numbers with possible parens
                    items = corefs.split('|')
                    for item in items:
                        #Get just the coref number
                        coref_num = int(''.join([s for s in item if s.isdigit()])) #the coreference number

                        #If the word itself is the entire entity
                        if item.startswith('(') and item.endswith(')'):
                            entities.append((word,coref_num)) #TODO: append more than just the string

                        #If we're beginning a new entity
                        #Note that there may still be other currently-incomplete entities
                        #Eg We can encounter an open paren without having closed the previous paren
                        elif item.startswith('('):
                            entity_status[coref_num] = True
                            entity_strings[coref_num] = word

                        #If we're ending an entity
                        #Note that other entities may still be open
                        elif item.endswith(')'):
                            entity_strings[coref_num] += "_{}".format(word)
                            entities.append((entity_strings[coref_num], coref_num)) #TODO: append more than just the string

                            #Clear the string
                            entity_strings[coref_num] = ""

                            #Stop picking up strings for this entity
                            entity_status[coref_num] = False
                else: #coref == '-'
                    #Add the current word to the string for all open entities
                    for coref_num in entity_status.keys():
                        if entity_status[coref_num] == True:
                            entity_strings[coref_num] += "_{}".format(word)
        return entities

def extract_entity_pairs(filename):
    entities = extract_entities(filename)
    #TODO: add more here

if __name__ == "__main__":
    entity_strings = extract_entities(SAMPLE_ANNOTATION)
    print(sorted(entity_strings, key=lambda X:X[1]))
