# -*- coding: utf-8 -*-

import collections


class DefaultLookupDict(dict):
    """Dictionary class with fall-back lookup with default value set in the constructor."""

    def __init__(self, default, d=None):
        if d:
            super(DefaultLookupDict, self).__init__(d)
        else:
            super(DefaultLookupDict, self).__init__()
        self._default = default

    def __getitem__(self, k):
        return self.get(k, self._default)


class Counter(collections.Counter):

    def discard(self, min_freq, unknown_token):
        freq = 0
        ret = Counter({})
        for token, count in self.items():
            if count < min_freq:
                freq += count
            else:
                ret[token] = count
        ret[unknown_token] = ret.get(unknown_token, 0) + freq
        return ret


def count_tokens(tokens, to_lower=False, counter=None):
    if to_lower:
        tokens = [t.lower() for t in tokens]

    if counter is None:
        return Counter(tokens)
    else:
        counter.update(tokens)
        return counter
