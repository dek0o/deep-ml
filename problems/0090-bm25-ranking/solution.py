import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
	n_docs = len(corpus)
	avgdl = sum([len(doc) for doc in corpus]) / n_docs
	doc_freq = Counter(word for doc in corpus for word in set(doc)) # term: count in corpus

	scores = []
	for doc in corpus:
		terms_fr = Counter(doc) # term: count in doc
		d_len = len(doc)

		scores.append(sum([np.log((n_docs + 1) / (doc_freq[t] + 1)) * (terms_fr[t] * (k1 + 1)) / (terms_fr[t] + k1 * (1 - b + b * (d_len / avgdl))) for t in query]))

	return np.round(scores,3)