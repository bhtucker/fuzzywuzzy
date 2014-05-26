#!/usr/bin/env python
# encoding: utf-8
"""
process.py

Copyright (c) 2011 Adam Cohen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import itertools

from . import fuzz
from . import utils


def extract(query, choices, processor=None, scorer=None, limit=5):
    """Find best matches in a list of choices, return a list of tuples
       containing the match and it's score.

    Arguments:
        query       -- an object representing the thing we want to find
        choices     -- a list of objects we are attempting to extract
                       values from
        scorer      -- f(OBJ, QUERY) --> INT. We will return the objects
                       with the highest score by default, we use
                       score.WRatio() and both OBJ and QUERY should be
                       strings
        processor   -- f(OBJ_A) --> OBJ_B, where the output is an input
                       to scorer for example, "processor = lambda x:
                       x[0]" would return the first element in a
                       collection x (of, say, strings) this would then
                       be used in the scoring collection by default, we
                       use utils.full_process()

    """
    if choices is None or len(choices) == 0:
        return []

    # default, turn whatever the choice is into a workable string
    if processor is None:
        processor = lambda x: utils.full_process(x)

    # default: wratio
    if scorer is None:
        scorer = fuzz.WRatio

    sl = list()

    for choice in choices:
        processed = processor(choice)
        score = scorer(query, processed)
        tuple = (choice, score)
        sl.append(tuple)

    sl.sort(key=lambda i: i[1], reverse=True)
    return sl[:limit]


def extractBests(query, choices, processor=None, scorer=None, score_cutoff=0, limit=5):
    """Find best matches above a score in a list of choices, return a
    list of tuples containing the match and it's score.

    Convenience method which returns the choices with best scores, see
    extract() for full arguments list

    Optional parameter: score_cutoff.
        If the choice has a score of less than or equal to score_cutoff
        it will not be included on result list

    """

    best_list = extract(query, choices, processor, scorer, limit)
    if len(best_list) > 0:
        return list(itertools.takewhile(lambda x: x[1] > score_cutoff, best_list))
    else:
        return []


def extractOne(query, choices, processor=None, scorer=None, score_cutoff=0):
    """Find the best match above a score in a list of choices, return a
    tuple containing the match and it's score if it's above the treshold
    or None.

    Convenience method which returns the single best choice, see
    extract() for full arguments list

    Optional parameter: score_cutoff.
        If the best choice has a score of less than or equal to
        score_cutoff we will return none (intuition: not a good enough
        match)

    """

    best_list = extract(query, choices, processor, scorer, limit=1)
    if len(best_list) > 0:
        best = best_list[0]
        if best[1] > score_cutoff:
            return best
        else:
            return None
    else:
        return None


"""
Sometimes an application might want to perform a lot of match extraction from the same set of choices.
A common solution in the loose string literature is to use a 'pruning' lower bound that's inexpensive to compute
and reduces the number of strings you'll actually call the scorer function with. A common pruning function
is to look at the n-gram signatures of the search pool and disqualify totally dissimilar strings.

In a large application, one might use something like Apache Solr, but you can easily achieve much
of the loose search performance gain using SciPy. The SciPy objects are sparse
COO (coordinate) matrices, and so are relatively lightweight and (I believe) can be pickled well.
"""

def getNGramVectorizerAndMatrix(choices, in_ngram_range = (3,3), in_analyzer='char',in_min_df = 1):
    """Return a (vectorizer,matrix) tuple with ngram respresentation of choices and vectorizer to 
    transform future queries. Defaults to 3-character ngrams; can be optionally set with in_ngram_range.
    """ 

    from sklearn.feature_extraction.text import CountVectorizer
    from scipy.sparse import dia_matrix
    import numpy as np

    #initialize the vectorizer
    ngram_vectorizer = CountVectorizer(analyzer=in_analyzer, ngram_range = in_ngram_range, min_df = in_min_df)

    #process the choices argument
    clean_choices = [utils.full_process(utils.asciidammit(x)) for x in choices]
    
    #create ngram features in the vectorizer for every ngram in the choices set
    ngram_vectorizer.fit(clean_choices)

    #create a matrix where each row is the ngram-feature representation of an element of choices
    choices_matrix = ngram_vectorizer.transform(clean_choices)

    #rescale these entries to all sum to 1 so that the dot-product measure doens't bias towards longer strings
    #(perhaps this should scale to have equal norm and not sum?)
    Y = choices_matrix.sum(axis=1)
    #make a diagonal matrix of the row sums for rescaling
    diag =dia_matrix(((1./Y.transpose()),[0]),shape=(len(Y),len(Y)))

    #overwrite choices_matrix with rescaled version
    choices_matrix = diag*choices_matrix

    return (ngram_vectorizer,choices_matrix)

def extractWithVectorizer(query, choices, choices_matrix, vectorizer, threshold=3,in_score_cutoff=94):
    """Given a query and choicelist (like normal extract()) and choice matrix and vectorizer, find
    choices with a ngram similarity above threshold (optional parameter, default to 3) and then 
    return the best match from that short list with a fuzzywuzzy score above in_score_cutoff (def. 94)
    """

    #ngram representation of query
    query_vector = vectorizer.transform([query])

    #array of dot products between query_vector and choice vectors in choices_matrix
    resultarray = np.array(choices_matrix.dot(query_vector.transpose()).todense())
    
    #array of indices of rows where dot product was above threshold
    choice_indices = np.where(resultarray > threshold)[0] 

    #list of strings from 'choices' at given indices
    short_list = [choices[i] for i in choice_indices]

    #extractOne to find best option from short list
    return extractOne(query,short_list,score_cutoff=in_score_cutoff)


