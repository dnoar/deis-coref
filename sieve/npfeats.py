import wikipedia
import preprocess

MALE = ['he', 'him', 'his']
FEMALE = ['she', 'her']
NEUTRAL = ['it', 'they', 'its', 'their']
DETERMINERS = ['the','a','an','this','these','that','those','my', 'your', 'his', 'her', 'its', 'our', 'their']


def check_gender(np,log):
	if np in log:
		return log[np]
	else:
		tempnp = np.split(' ')
		if len(tempnp) == 1:
			if np in MALE:
				return 'male'
			if np in FEMALE:
				return 'female'
			if np in NEUTRAL:
				return 'neutral'
		if tempnp[0] in DETERMINERS:
			tempnp = ' '.join(tempnp[1:])
		summary = []
		try:
			summary = wikipedia.summary(tempnp).split(' ')
		except:
			try:
				queries = wikipedia.search(tempnp)
				summary = wikipedia.summary(queries[1]).split(' ') #to avoid disambiguation errors
			except:
				try:
					summary = wikipedia.summary(np).split(' ')
				except:
					try:
						queries = wikipedia.search(np)
						summary = wikipedia.summary(queries[1]).split(' ') #to avoid disambiguation errors
					except:
						pass

		if summary != []:
			male = 0
			female = 0
			neutral = 0

			for pronoun in MALE:
				male += summary.count(pronoun)
			for pronoun in FEMALE:
				female += summary.count(pronoun)
			for pronoun in NEUTRAL:
				neutral += summary.count(pronoun)

			if male > female and male >= neutral:
				return 'male'
			if female > male and female >= neutral:
				return 'female'
			else:
				return 'neutral'
	return 'undetermined'

def check_plurality(np,log):
	if np in log:
		return log[np]
	else:
		tempnp = np.split(' ')
		if 'and' in tempnp:
			return 'plural-and'
		if np[0] in DETERMINERS:
			if np[0] in ['these','those']:
				return 'plural-det'
			if np[0] in ['the','a','an','this','that']:
				return 'single-det'
		page = None
		try:
			page = wikipedia.page(np)
		except:
			try:
				queries = wikipedia.search(np)
				page = wikipedia.page(queries[1]) #to avoid disambiguation errors
			except:
				pass

		if page != None:
			summary = page.summary.split(' ')

			single = 0
			plural = 0

			single += summary.count('is')
			plural += summary.count('are')

			if single > plural:
				if page.title[-1] == 's' and np[-1] != 's':
					return 'single-titlemismatch'
				return 'single'
			if plural > single:
				if page.title[-1] != 's' and np[-1] == 's':
					return 'plural-titlemismatch'
				return 'plural'

	return 'undetermined'


def quick_check(np):
	return (check_gender(np,{}),check_plurality(np,{}))

def quick_check_logs(np,log_g,log_p):
	log_g[np.replace('_',' ').lower()] = check_gender(np.replace('_',' ').lower(),log_g)
	log_p[np.replace('_',' ').lower()] = check_plurality(np.replace('_',' ').lower(),log_p)
	return (log_g[np.replace('_',' ').lower()],log_p[np.replace('_',' ').lower()],log_g,log_p)

def load_logs():
	log_g = {}
	log_p = {}
	with open('genders.txt',encoding='utf8') as genders:
		for line in genders:
			log_g[line.split('|SPLIT|')[0]] = line.split('|SPLIT|')[1]
	with open('plurality.txt',encoding='utf8') as plurality:
		for line in plurality:
			log_p[line.split('|SPLIT|')[0]] = line.split('|SPLIT|')[1]
	return log_g,log_p

def save_logs(log_g,log_p):
	with open('genders.txt','w',encoding='utf8') as genders:
		for np in log_g:
			genders.write(np+'|SPLIT|'+log_g[np]+'\n')
	with open('plurality.txt','w',encoding='utf8') as plurality:
		for np in log_p:
			plurality.write(np+'|SPLIT|'+log_p[np]+'\n')

if __name__ == '__main__':
	#log_g,log_p = load_logs()
	feats = 'coref.feat'
	sent_nps = preprocess.get_nps(feats)
	log_g = {}
	log_p = {}
	for np_list in sent_nps:
		for np in sent_nps[np_list]:
			temp = np.replace('_',' ')
			print(temp)
			log_g[temp.lower()] = check_gender(temp.lower(),log_g)
			log_p[temp.lower()] = check_plurality(temp.lower(),log_p)
	save_logs(log_g,log_p)
