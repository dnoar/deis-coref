from nltk.tree import Tree
import preprocess

def save_trees():
	print('saving trees')
	trees = preprocess.get_trees('coref.feat') #longorshort
	counter = 0
	with open('trees.txt','w',encoding='utf8') as tree_file: #longorshort
		for tree in trees:
			tree_file.write(str(tree)+'|SPLIT|'+str(trees[tree])+'\n')
			counter += 1

def open_trees(load):
	easy_trees = {}

	counter = 0
	if load == False:
		save_trees()
	print('opening trees')
	with open('trees.txt',encoding='utf8') as tree_file: #longorshort
		for row in tree_file:
			'easy_trees[doc_ID][part_num][sent_num] = {tree : string tree, name: (doc, part, sent)}'
			try: #if doc part already in easy trees
				easy_trees[row.split('|SPLIT|')[0].split('\'')[1]][row.split('|SPLIT|')[0].split('\'')[3]][row.split('|SPLIT|')[0].split('\'')[5]] = {'tree':row.split('|SPLIT|')[1],'name':row.split('|SPLIT|')[0]}
			except:
				try: #if doc already in easy trees
					easy_trees[row.split('|SPLIT|')[0].split('\'')[1]][row.split('|SPLIT|')[0].split('\'')[3]] = {}
					easy_trees[row.split('|SPLIT|')[0].split('\'')[1]][row.split('|SPLIT|')[0].split('\'')[3]][row.split('|SPLIT|')[0].split('\'')[5]] = {'tree':row.split('|SPLIT|')[1],'name':row.split('|SPLIT|')[0]}
				except: #if no relevant keys in easy trees
					easy_trees[row.split('|SPLIT|')[0].split('\'')[1]] = {}
					easy_trees[row.split('|SPLIT|')[0].split('\'')[1]][row.split('|SPLIT|')[0].split('\'')[3]] = {}
					easy_trees[row.split('|SPLIT|')[0].split('\'')[1]][row.split('|SPLIT|')[0].split('\'')[3]][row.split('|SPLIT|')[0].split('\'')[5]] = {'tree':row.split('|SPLIT|')[1],'name':row.split('|SPLIT|')[0]}

			counter += 1

	easy_trees = convert_trees(easy_trees)

	return easy_trees

def convert_trees(trees):
	print('converting trees')
	counter = 0

	for treeKey in trees.keys():
		for partKey in trees[treeKey].keys():
			for sentKey in trees[treeKey][partKey].keys():
				trees[treeKey][partKey][sentKey]['nltk'] = Tree.fromstring(trees[treeKey][partKey][sentKey]['tree'])

		counter += 1
	return trees

def find_pronouns(tree):
	tree = tree['nltk']
	index = 0
	pronouns = []
	for (word,pronoun) in tree.pos():
		if pronoun == 'PRP':
			pronouns.append(tree.leaf_treeposition(index))
		index += 1
	return pronouns

def find_dominating_NP(path,tree):
	subtree = tree['nltk']
	dominating = -1
	for i in range(len(path)-1):
		subtree = subtree[path[i]]
		if subtree.label() == 'NP' or subtree.label() == 'S':
			dominating = i
	if dominating > -1:
		return tuple(list(path)[:dominating+1])
	else:
		return ()

def find_all_NPs(tree):
	tree = tree['nltk']
	paths = tree.treepositions()
	nps = []

	for path in paths:
		subtree = tree
		for step in path:
			subtree = subtree[step]
		try:
			if subtree.label() == 'NP':
				nps.append(path)
		except:
			pass

	return nps

def find_past_trees(trees,tree):
	doc_key = tree['name'].split('\'')[1]
	part_key = int(tree['name'].split('\'')[3])
	sent_key = tree['name'].split('\'')[5]

	past = []
	
	for i in range(int(sent_key)-1,-1,-1):
		try:
			past.append(trees[doc_key][str(part_key)][str(i)])
		except:
			part_key -= 1
			past.append(trees[doc_key][str(part_key)][str(i)])

	return past


def check_proposal(np,tree):
	'add some checks in here so it actually does something'
	return True

def check_current_nps(x_path, p, fulltree, nps, iteration):
	tree = fulltree['nltk']
	potential_nps = []
	if iteration < 2:
		potential_nps = [np for np in nps if np[:len(x_path)] == x_path and np != x_path]
	else:
		potential_nps = [np for np in nps if np[:len(x_path)] == x_path]
	potential_nps.sort(key = lambda x: (len(x),x[-1]))

	if iteration < 2:
		for np in potential_nps:
			if len(np) > len(x_path) and len(p) > len(x_path):
				if np[len(x_path)] < p[len(x_path)]:
					subtree = tree
					count = 0
					for i in range(len(np)):
						subtree = subtree[np[i]]
						if i > len(x_path):
							try:
								if subtree.label() == 'NP' or subtree.label() == 'S':
									count += 1
							except:
								pass
					if count > 1:
						if check_proposal(np,tree):
							return [np,fulltree]
	else:
		x_subtree = tree
		for step in x_path:
			x_subtree = x_subtree[step]
		if x_subtree.label() == 'NP':
			for np in potential_nps:
				subtree = tree
				for i in range(len(np)):
					subtree = subtree[np[i]]	
				if check_proposal(np,tree):
					return [np,fulltree]
		else:
			for np in potential_nps:
				subtree = tree
				for i in range(len(np)):
					subtree = subtree[np[i]]
					if subtree.label() == 'NP' or subtree.label() == 'S' and i == len(np)-1:
						if subtree.label() == 'NP':
							if check_proposal(np,tree):
								return [np,fulltree]

	return []

def check_past_nps(past_trees):
	for tree in past_trees:
		potential_nps = find_all_NPs[tree]
		potential_nps.sort(key = lambda x: (len(x),x[-1]))

		for np in potential_nps:
			if check_proposal(np,tree):
				return [np,tree]

	return []

def hobbs(node, tree, trees, iteration):
	x = find_dominating_NP(node,tree)
	proposal = []
	proposal = check_current_nps(x,node,tree,find_all_NPs(tree),iteration)
	if proposal == None:
		if len(x) > 0:
			iteration += 1
			hobbs(x,tree,trees,iteration)
		else:
			proposal = check_past_nps(find_past_trees(tree,trees))
	return proposal

def link_proposals(all_proposals,proposed,tree,prp_path):
	'takes the proposed ties and saves them to the pronoun'
	prp_name = tree['name']+'_'+str(prp_path)
	all_proposals[tree['name']+'__'+str(np_to_leaves(list(prp_path[:-1]),tree['nltk']))] = proposed[1]['name']+'__'+str(np_to_leaves(proposed[0],proposed[1]['nltk']))
	return all_proposals

def np_to_leaves(path,tree):
	'takes a path list and a nltk tree, returns leaf indexes dominated by path node'
	np_leaves = []
	subtree = tree

	for i in range(len(tree.leaves())):
		leaf_path = tree.leaf_treeposition(i)
		if list(path) == list(leaf_path)[:len(path)]:
			np_leaves.append(i)

	return str(np_leaves)

def link_chains(all_proposals):
	counter = 0
	chains = {}
	node_locations = {}
	for prop in all_proposals:
		if prop in node_locations:
			if all_proposals[prop] not in node_locations:
				chains[node_locations[prop]].append(all_proposals[prop])
		elif all_proposals[prop] in node_locations:
			chains[node_locations[all_proposals[prop]]].append(prop)
		else:
			chains[counter] = [prop, all_proposals[prop]]
			node_locations[prop] = counter
			node_locations[all_proposals[prop]] = counter
			counter += 1
	return chains

def chains_to_feat(chains):
	feats = open('coref.feat',encoding='utf8') #longorshort

	labels = {}
	for chain in chains:
		for np in chains[chain]:
			name = np.split('__')[0]
			leaves = np.split('__')[1].strip('[]')
			all_leaves = leaves.split(', ')


			for i in range(len(all_leaves)):
				if len(all_leaves) > 2:
					if i != 0 or i != len(all_leaves)-1:
						all_leaves[i] = '-'
				leafnode = name[:-1]+', \''+all_leaves[i]+'\')'
				chainlabel = str(chain)
				if i == 0:
					chainlabel = '(' + chainlabel
				if i == len(all_leaves)-1:
					chainlabel += ')'
				labels[leafnode] = chainlabel

	with open('output.txt','w',encoding='utf8') as out:
		for line in feats:
			outputline = line
			if line != 'doc_id,part_num,sent_num,word_num,word,pos,parse_bit,pred_lemma,pred_frame_id,sense,speaker,ne,corefs':
				doc = line.split(',')[0]
				part = line.split(',')[1]
				sent = line.split(',')[2]
				word = line.split(',')[3]
				linename = str((doc,part,sent,word))
				

				if linename in labels:
					row = line.split(',')
					row[-1] = labels[linename]
					outputline = ','.join(row) + '\n'
				else:
					row = line.split(',')
					row[-1] = '-'
					outputline = ','.join(row) + '\n'

			out.write(outputline)
	feats.close()



if __name__ == '__main__':
	trees = open_trees(False) #false forces a new save of trees, true just loads from file.
	all_proposals = {}
	counter = 1
	for doc in trees:
		for part in trees[doc]:
			for sent in trees[doc][part]:
				print('sentence '+str(counter))
				counter += 1
				pronouns = find_pronouns(trees[doc][part][sent])

				for i in range(len(pronouns)):
					proposals = hobbs(find_dominating_NP(pronouns[i],trees[doc][part][sent]),trees[doc][part][sent],trees,1)

					if proposals != []:
						all_proposals = link_proposals(all_proposals,proposals,trees[doc][part][sent],pronouns[i])

	chains = link_chains(all_proposals)
	chains_to_feat(chains)

