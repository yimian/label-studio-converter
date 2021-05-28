# -*- coding: utf-8 -*-

# @File    : utils.py
# @Date    : 2021-05-28
# @Author  : skym
from refo.virtualmachine import MatchType
from refo import Predicate, Star, Any, Group


def build_pattern(term):
    phrases = [Phrase(p) for p in term.split()]
    pattern = phrases[0]
    for p in phrases[1:]:
        pattern += Star(Any(), greedy=False) + p
    return pattern


def build_fuzzy_pattern(term):
    phrases = [Group(FuzzyPhrase(p), '_%s' % i) for i, p in enumerate(term.split())]
    pattern = phrases[0]
    for p in phrases[1:]:
        pattern += Star(Any(), greedy=False) + p
    return pattern


class Phrase(Predicate):
    def __init__(self, x):
        super().__init__(self.match)
        self.x = x
        self.arg = x

    def match(self, y):
        if not isinstance(y, (tuple, list)):
            y = [y]
        y = ''.join(y)
        if self.x == y:
            return MatchType.FullMatch
        elif self.x.startswith(y):
            return MatchType.PartialMatch
        else:
            return MatchType.NotMatch


class FuzzyPhrase(Predicate):
    '''
    模糊匹配。当前模糊匹配存在下面问题：

    "超级" "超级" "好闻" 匹配"超级好"时，会返回"超级超级好闻"

    主要是存在前后2个token是一样。需要通过后续处理措施解决

    '''

    def __init__(self, x):
        super().__init__(self.match)
        self.x = x
        self.arg = x

    def match(self, y):
        if not isinstance(y, (tuple, list)):
            y = [y]
        y = ''.join(y)
        if self.x in y:
            return MatchType.FullMatch
        elif self.is_overlapped(y):
            return MatchType.PartialMatch
        else:
            return MatchType.NotMatch

    def is_overlapped(self, y):
        i = len(self.x) - 1
        while i > 0:
            if y.endswith(self.x[:i]):
                return True
            i -= 1
        return False


def get_term_from_match(tokens, m, term):
    ret = []
    c = len(term.split())
    new_term = ''
    last_s = 0
    for i in range(c):
        s, e = m.group('_%s' % i)
        # 排除"超级好闻"匹配到"超级超级好闻"
        cur_term = ''
        for tok in reversed(tokens[s:e]):
            cur_term = tok + cur_term
            if term in cur_term:
                new_term = cur_term
                return new_term
        # 判断前后2个是否连续，是否需要加空格
        if last_s == s:
            new_term += cur_term
        elif new_term != '':
            new_term += ' ' + cur_term
        else:
            new_term = cur_term
        last_s = e
    return new_term
