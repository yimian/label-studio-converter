# -*- coding: utf-8 -*-

# @File    : loader.py
# @Date    : 2021-05-27
# @Author  : skym
import string
import random
import os
import json
import copy

from ainlp.text_utils import TokenMatcher
from loguru import logger

from refo import search
from label_studio_converter.custom.utils import (
    get_term_from_match,
    build_pattern,
    build_fuzzy_pattern)


class ABSALoader:
    """"""

    def __init__(self,
                 lang="cn",
                 level_columns=['nlp_level_1', 'nlp_level_2'],
                 keep_columns=['nlp_unique_id'],
                 label_columns=['nlp_matched_key', 'nlp_summary'],
                 choice_column='nlp_review_rating',
                 text_column='text'):

        """
            convert absa data from csv to label-studio YM_ABSA format
        """
        self.clause_delimiter = (
            [",", ".", "?", "!", ";", "~", "，", "？", "！", "；", "…"] if lang != "cn" else None
        )
        self.err_entries = []

        self.level_columns = level_columns
        self.keep_columns = keep_columns
        self.aspect_column = label_columns[0]
        self.opinion_column = label_columns[1]
        self.choice_column = choice_column
        self.text_column = text_column

        self.visited = set()
        self.lang = lang

    def _gen_result(self, item, matcher):
        """每条op只能生成0-1个情感摘要, 特征情感词不在同一个分句不生成, 如果有多条, 取特征情感词位置最接近的"""

        result_item, convert_result_item = [], []
        for as_term in item[self.aspect_column]:
            # 为每一个特征词寻找最佳的情感摘要匹配, 如果一个都没有匹配到返回首个特征词的匹配
            found_matches = []

            for op_term in item[self.opinion_column]:
                op_term = str(op_term)
                # print(f'{as_term} | {op_term}')

                m = matcher.search_co_occurance(
                    as_term.replace('  ', ' '), op_term.replace('  ', ' ')
                )
                if m:
                    aspect_tok_pos, op_tok_pos = m.group('kw1'), m.group('kw2')
                    found_matches.append((m, abs(op_tok_pos[0] - aspect_tok_pos[0])))

            if len(found_matches) > 0:
                m, _ = min(found_matches, key=lambda x: x[1])
                cover_tok_pos, aspect_tok_pos, op_tok_pos = (
                    m.group(),
                    m.group('kw1'),
                    m.group('kw2'),
                )
                # 只有特征词与情感词在一个分句则认为summary可以识别
                tok_start, tok_end = cover_tok_pos[0], cover_tok_pos[1]
                if (
                        matcher.clause_index[aspect_tok_pos[0]]
                        == matcher.clause_index[op_tok_pos[0]]
                ):
                    tmp = matcher.pos_index[tok_start: tok_end + 1]
                    start, end = tmp[0], tmp[-1]
                    shot_text = matcher.text[start:end]

                    result_id = self._gen_id()
                    r, cr = self._gen_aspect(matcher=matcher, tok_range=aspect_tok_pos)
                    result_item.extend(r)
                    convert_result_item.extend(cr)

                    result_item.append(
                        {
                            'from_name': 'union_labeltype',
                            'to_name': 'text',
                            'source': '$text',
                            'id': result_id,
                            'type': 'absalabels',
                            'value': {
                                'end': end + tok_end - 1 if self.lang == "cn" else end,
                                'start': start + tok_start if self.lang == "cn" else start,
                                'text': shot_text,
                                'absalabels': ['OpinionSummary'],
                            },
                        }
                    )
                    convert_result_item.append(
                        {
                            'from_name': 'union_labeltype',
                            'to_name': 'text',
                            'source': '$text',
                            'id': result_id,
                            'type': 'absalabels',
                            'value': {
                                'end': end,
                                'start': start,
                                'text': shot_text,
                                'absalabels': ['OpinionSummary'],
                            },
                        }
                    )

        if len(result_item) == 0:
            # print('没有情感摘要{}:{}'.format(self.answer['question_id'], item['aspectSubtype']) )
            # 如果一个摘要都没有找到, 列出首个特征词的结果
            return self._gen_aspect(kw=item[self.aspect_column][0],
                                    matcher=matcher)

        return result_item, convert_result_item

    def _gen_aspect(self, matcher, kw=None, tok_range=None):
        """
        根据特征词生成 yicrowds absa 标注格式 result item
        :param label_type:
        :param kw:
        :param tok_range: 如果传入 tok_range 直接根据 tok_range 生成结果
        :return:
        """
        if tok_range is None:
            kw_range = matcher.search(kw.replace('  ', ' '))

            if kw_range is None:
                return [], []

            tok_start, tok_end = kw_range.start(), kw_range.end()
        else:
            tok_start, tok_end = tok_range

        tmp = matcher.pos_index[tok_start: tok_end + 1]
        start, end = tmp[0], tmp[-1]
        shot_text = matcher.text[start:end]

        result_id = self._gen_id()
        result_item, convert_result_item = (
            [
                {
                    'from_name': 'union_labeltype',
                    'to_name': 'text',
                    'source': '$text',
                    'id': result_id,
                    'type': 'absalabels',
                    'value': {
                        'end': end + tok_end - 1 if self.lang == "cn" else end,
                        'start': start + tok_start if self.lang == "cn" else start,
                        'text': shot_text,
                        'absalabels': ['AspectTerm'],
                    },
                }
            ],
            [
                {
                    'from_name': 'union_labeltype',
                    'to_name': 'text',
                    'source': '$text',
                    'id': result_id,
                    'type': 'absalabels',
                    'value': {
                        'end': end,
                        'start': start,
                        'text': shot_text,
                        'absalabels': ['AspectTerm'],
                    },
                }
            ],
        )

        return result_item, convert_result_item

    def _init_polar_result(self, item):
        """转换极性标注结果"""
        choice = {
            'from_name': 'union_choice',
            'to_name': 'original_text',
            'id': self._gen_id(),
            'type': 'choices',
            'value': {'choices': [str(item[self.choice_column])]},
        }

        return [choice], [choice]

    def _load_item(self, item, matcher):
        result, convert_result = self._init_polar_result(item)

        result_item, convert_result_item = self._gen_result(item, matcher)
        convert_result.extend(convert_result_item)
        result.extend(result_item)

        return result, convert_result

    def _preprocess_item(self, item):
        new_item = copy.deepcopy(item)
        aspect_term, opinion_term = new_item[self.aspect_column], new_item[self.opinion_column]
        new_item[self.aspect_column] = aspect_term.strip().split(',') if isinstance(aspect_term, str) else []
        new_item[self.opinion_column] = opinion_term.strip().split(',') if isinstance(opinion_term, str) else []
        return new_item

    def load(self, item):
        """convert line of nlp csv to a label-studio absa task"""
        item = self._preprocess_item(item)
        matcher = TokenMatcher(item[self.text_column].lower(), clause_delimiter=self.clause_delimiter)

        result, convert_result = self._load_item(item, matcher)
        if len(result) == 1:
            # 重新查找
            try:
                has_error, new_item, reason = self._check_item(item, tokens=matcher.tokens)
                if has_error and new_item is not None:
                    result, convert_result = self._load_item(new_item, matcher)
            except Exception as e:
                logger.error(repr(e))

        if len(result) == 1:
            return False, {}

        else:
            return True, {
                'data': self._prepare_data(item, matcher.tokens),
                'completions': [
                    {
                        'result': result,
                        'convert_result': convert_result,
                    }
                ],
            }

    def _prepare_data(self, item, tokens):
        data = {
            'text': ' '.join(tokens) if self.lang == "cn" else item[self.text_column],
            'original_text': item[self.text_column],
            'nlp_level': '#'.join([item[level] for level in self.level_columns])
        }

        for col in self.keep_columns:
            data.update({col: item[col]})
        return data

    def _gen_id(self):
        return ''.join(random.sample(string.ascii_letters + '-_' + string.digits, 10))

    def _check_item(self, op, tokens):
        new_op = copy.deepcopy(op)
        has_error, has_fatal = False, False
        reason = {}
        label_content = [(item, new_op[item]) for item in [self.aspect_column, self.opinion_column]]
        for key, detail in label_content:
            for i, term in enumerate(detail):
                pat = build_pattern(term)
                m = search(pat, tokens)
                if not m:
                    has_error = True
                    fuzzy_pat = build_fuzzy_pattern(term)
                    m = search(fuzzy_pat, tokens)
                    if not m:
                        reason[key] = f'{term} not found'
                        has_fatal = True
                    else:
                        new_term = get_term_from_match(tokens, m, term)
                        if term != new_term:
                            new_op[key] = [new_term]
                            reason[key] = f'{term} change to {new_term}'
                        else:
                            return False, None, None
        if has_fatal:
            return True, None, reason
        if has_error:
            return True, new_op, reason
        return False, None, None
