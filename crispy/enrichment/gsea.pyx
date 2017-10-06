import random
import operator
import numpy as np


def gsea(dataset, signature, permutations=100):
    cdef int i

    # Sort data-set by values
    _dataset = list(zip(*sorted(dataset.items(), key=operator.itemgetter(1), reverse=False)))
    genes, expression = _dataset[0], _dataset[1]

    # Signature overlapping the data-set
    _signature = set(signature).intersection(genes)

    # Check signature overlap
    e_score, p_value = np.NaN, np.NaN
    hits, running_hit = [], []
    if len(_signature) != 0:

        # ---- Calculate signature enrichment score
        n, sig_size = len(genes), len(_signature)
        nh = n - sig_size
        nr = sum([abs(dataset[g]) for g in _signature])

        e_score = __es(genes, expression, _signature, nr, nh, n, hits, running_hit)

        # ---- Calculate statistical enrichment
        # Generate random signatures sampled from the data-set genes
        count = 0

        for i in xrange(permutations):
            r_signature = random.sample(genes, sig_size)

            r_nr = sum([abs(dataset[g]) for g in r_signature])

            r_es = __es(genes, expression, r_signature, r_nr, nh, n)

            if (r_es >= e_score >= 0) or (r_es <= e_score < 0):
                count += 1

        # If no permutation was above the Enrichment score the p-value is lower than 1 divided by the number of permutations
        p_value = 1 / permutations if count == 0 else count / permutations

    return e_score, p_value, hits, running_hit


def __es(genes, expression, signature, nr, nh, n, hits=None, running_hit=None):
    cdef int i
    cdef double hit, miss, es, r

    hit, miss, es, r = 0, 0, 0, 0
    for i in xrange(n):
        if genes[i] in signature:
            hit += abs(expression[i]) / nr

            if hits is not None:
                hits.append(1)

        else:
            miss += 1 / nh

            if hits is not None:
                hits.append(0)

        r = hit - miss

        if running_hit is not None:
            running_hit.append(r)

        if abs(r) > abs(es):
            es = r

    return es