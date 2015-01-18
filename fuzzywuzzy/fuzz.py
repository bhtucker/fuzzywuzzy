#!/usr/bin/env python
# encoding: utf-8
"""
fuzz.py

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
from __future__ import unicode_literals

try:
    from StringMatcher import StringMatcher as SequenceMatcher
except:
    from difflib import SequenceMatcher

from . import utils


###########################
# Basic Scoring Functions #
###########################


def ratio(s1, s2):

    if s1 is None:
        raise TypeError("s1 is None")
    if s2 is None:
        raise TypeError("s2 is None")
    s1, s2 = utils.make_type_consistent(s1, s2)
    if len(s1) == 0 or len(s2) == 0:
        return 0

    m = SequenceMatcher(None, s1, s2)
    return utils.intr(100 * m.ratio())


# todo: skip duplicate indexes for a little more speed
def partial_ratio(s1, s2):

    if s1 is None:
        raise TypeError("s1 is None")
    if s2 is None:
        raise TypeError("s2 is None")
    s1, s2 = utils.make_type_consistent(s1, s2)
    if len(s1) == 0 or len(s2) == 0:
        return 0

    if len(s1) <= len(s2):
        shorter = s1
        longer = s2
    else:
        shorter = s2
        longer = s1

    m = SequenceMatcher(None, shorter, longer)
    blocks = m.get_matching_blocks()

    # each block represents a sequence of matching characters in a string
    # of the form (idx_1, idx_2, len)
    # the best partial match will block align with at least one of those blocks
    #   e.g. shorter = "abcd", longer = XXXbcdeEEE
    #   block = (1,3,3)
    #   best score === ratio("abcd", "Xbcd")
    scores = []
    for block in blocks:
        long_start = block[1] - block[0] if (block[1] - block[0]) > 0 else 0
        long_end = long_start + len(shorter)
        long_substr = longer[long_start:long_end]

        m2 = SequenceMatcher(None, shorter, long_substr)
        r = m2.ratio()
        if r > .995:
            return 100
        else:
            scores.append(r)

    return int(100 * max(scores))


##############################
# Advanced Scoring Functions #
##############################

# Sorted Token
#   find all alphanumeric tokens in the string
#   sort those tokens and take ratio of resulting joined strings
#   controls for unordered string elements
def _token_sort(s1, s2, partial=True, force_ascii=True):

    if s1 is None:
        raise TypeError("s1 is None")
    if s2 is None:
        raise TypeError("s2 is None")

    # pull tokens
    tokens1 = utils.full_process(s1, force_ascii=force_ascii).split()
    tokens2 = utils.full_process(s2, force_ascii=force_ascii).split()

    # sort tokens and join
    sorted1 = " ".join(sorted(tokens1))
    sorted2 = " ".join(sorted(tokens2))

    sorted1 = sorted1.strip()
    sorted2 = sorted2.strip()

    if partial:
        return partial_ratio(sorted1, sorted2)
    else:
        return ratio(sorted1, sorted2)


def token_sort_ratio(s1, s2, force_ascii=True):
    return _token_sort(s1, s2, partial=False, force_ascii=force_ascii)


def partial_token_sort_ratio(s1, s2, force_ascii=True):
    return _token_sort(s1, s2, partial=True, force_ascii=force_ascii)


# TODO: numerics

###################
# Combination API #
###################

# q is for quick
def QRatio(s1, s2, force_ascii=True):

    p1 = utils.full_process(s1, force_ascii=force_ascii)
    p2 = utils.full_process(s2, force_ascii=force_ascii)

    if not utils.validate_string(p1):
        return 0
    if not utils.validate_string(p2):
        return 0

    return ratio(p1, p2)


def UQRatio(s1, s2):
    return QRatio(s1, s2, force_ascii=False)


def WRatio(s1, s2):

    s = [s1, s2]
    unbase_scale = .95

    base = ratio(s1, s2)

    lens = map(len, s)
    len_ratio = float(max(lens)) / min(lens)

    if len_ratio >= 1.5:
        try_partial = True
        if min(lens) > 6 and len_ratio > 3:
            # this is an exponential curve fit to their two-case values
            partial_scale = (0.988278) * (0.939527 ** len_ratio)
        else:
            partial_scale = .9

    else:
        try_partial = False

    tokens = map(lambda x: x.split(), s)

    # sort tokens and join
    sorted_tokens = map(lambda x: " ".join(sorted(x)).strip(), tokens)

    tokensets = map(set, tokens)

    token_tuples = [(tokensets[i], tokens[i]) for i in range(len(tokens))]

    # if the number of the tokens in the set is significantly different than the number of tokens in the list,
    # we should penalize
    set_penalties = [1, 1]
    for i in range(2):
        token_tuple = token_tuples[i]

        if len(token_tuple[0]) == 1 and len(token_tuple[1]) < 4 and len(token_tuple[1]) > 1:
            set_penalties[i] = .8
        else:
            set_penalties[i] = 1. - (.05 * (len(token_tuple[1]) - len(token_tuple[0])))

    set_penalty = min(set_penalties)
    #print set_penalty

    if try_partial:
        #print 'partial'
        partial = _partial_ratio(s) * partial_scale
        ptsor = _partial_ratio(sorted_tokens) \
            * unbase_scale * partial_scale
        ptser = _token_set(tokensets) \
            * unbase_scale * partial_scale * set_penalty

        #print base, partial, ptsor, ptser
        return int(max(base, partial, ptsor, ptser))

    else:
        #print 'non partial'
        tsor = ratio(sorted_tokens[0], sorted_tokens[1]) * unbase_scale

        tser = _token_set(tokensets, partial=False) \
            * unbase_scale * set_penalty

        #print base, tsor, tser
        return int(max(base, tsor, tser))


def _partial_ratio(sorted_tokens):

    assert all(sorted_tokens)
    s1, s2 = sorted_tokens[0], sorted_tokens[1]

    s1, s2 = utils.make_type_consistent(s1, s2)

    if len(s1) == 0 or len(s2) == 0:
        return 0

    if len(s1) <= len(s2):
        shorter = s1
        longer = s2
    else:
        shorter = s2
        longer = s1

    m = SequenceMatcher(None, shorter, longer)
    blocks = m.get_matching_blocks()

    # each block represents a sequence of matching characters in a string
    # of the form (idx_1, idx_2, len)
    # the best partial match will block align with at least one of those blocks
    #   e.g. shorter = "abcd", longer = XXXbcdeEEE
    #   block = (1,3,3)
    #   best score === ratio("abcd", "Xbcd")

    start_set = set()

    scores = []
    for block in blocks:
        long_start = block[1] - block[0] if (block[1] - block[0]) > 0 else 0

        if long_start in start_set:
            continue
        else:
            start_set.add(long_start)

        long_end = long_start + len(shorter)
        long_substr = longer[long_start:long_end]

        m2 = SequenceMatcher(None, shorter, long_substr)
        r = m2.ratio()
        if r > .995:
            return 100
        else:
            scores.append(r)

    return int(100 * max(scores))


def _token_set(tokens, partial=True, force_ascii=True):
    """
    Token Set
      find all alphanumeric tokens in each string...treat them as a set

      construct two strings of the form
          <sorted_intersection><sorted_remainder>
      take ratios of those two strings
      controls for unordered partial matches
    """

    assert all([isinstance(x, type(set())) for x in tokens]), \
        'expecting a sequence of sets for first arg tokens'

    intersection = " ".join(sorted(tokens[0].intersection(tokens[1]))).strip()
    diffs = [
        " ".join(sorted(x)).strip()
        for x in (
            tokens[0].difference(tokens[1]),
            tokens[1].difference(tokens[0])
        )
    ]

    combined_1to2 = intersection + " " + diffs[0]
    combined_2to1 = intersection + " " + diffs[1]

    pairwise = [
        ratio(intersection, combined_1to2),
        ratio(intersection, combined_2to1),
        ratio(combined_1to2, intersection)
    ]

    return max(pairwise)
