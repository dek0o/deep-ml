import numpy as np
from collections import Counter

def compute_tf_idf(corpus, query):
	"""
	Compute TF-IDF scores for a query against a corpus of documents.
    
	:param corpus: List of documents, where each document is a list of words
	:param query: List of words in the query
	:return: List of lists containing TF-IDF scores for the query words in each document
	"""
	document_frequency = {} # word: num of docs containing
	for doc in corpus:
		seen = set()
		for word in doc:
			if word not in seen:
				document_frequency[word] = document_frequency.get(word, 0) + 1
	
	inverted_document_frequency = {} # word: idf
	total_docs = len(corpus)
	for word, doc_fr in document_frequency.items():
		inverted_document_frequency[word] = np.log((total_docs + 1) / (doc_fr + 1)) + 1
	
	result = []
	for doc in corpus:
		doc_len = len(doc)
		words_count = Counter(doc)
		doc_result = []
		for query_word in query:
			term_frequency = words_count.get(query_word, 0) / doc_len
			doc_result.append(term_frequency * inverted_document_frequency.get(query_word, 0))
		result.append(doc_result)
	
	return result



