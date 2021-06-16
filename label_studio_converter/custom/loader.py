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
from ainlp import TextProcessor
from loguru import logger

from refo import search
from label_studio_converter.custom.utils import (
    get_term_from_match,
    build_pattern,
    build_fuzzy_pattern)


class ABSALoader:
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
        self.processor = TextProcessor()
        self.level_columns = level_columns
        self.keep_columns = keep_columns
        self.label_columns = label_columns
        self.aspect_column = label_columns[0]
        self.opinion_column = label_columns[1]
        self.choice_column = choice_column
        self.text_column = text_column

        self.visited = set()
        self.lang = lang
        self.matcher = None

    def _gen_result(self, item):
        """每条 op 只能生成 0-1个情感摘要, 特征情感词不在同一个分句不生成, 如果有多条, 取特征情感词位置最接近的"""
        result_item, convert_result_item = [], []
        for as_term in item[self.aspect_column]:
            # 为每一个特征词寻找最佳的情感摘要匹配, 如果一个都没有匹配到返回首个特征词的匹配
            found_matches = []

            for op_term in item[self.opinion_column]:

                op_term = str(op_term)

                m = self.matcher.search_co_occurance(
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
                        self.matcher.clause_index[aspect_tok_pos[0]]
                        == self.matcher.clause_index[op_tok_pos[0]]
                ):
                    tmp = self.matcher.pos_index[tok_start: tok_end + 1]
                    start, end = tmp[0], tmp[-1]
                    shot_text = self.matcher.text[start:end]

                    result_id = self._gen_id()
                    r, cr = self._gen_aspect(tok_range=aspect_tok_pos)
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
            return self._gen_aspect(kw=item[self.aspect_column][0])

        return result_item, convert_result_item

    def _gen_aspect(self, kw=None, tok_range=None):
        """
        根据特征词生成 yicrowds absa 标注格式 result item
        :param label_type:
        :param kw:
        :param tok_range: 如果传入 tok_range 直接根据 tok_range 生成结果
        :return:
        """
        if tok_range is None:
            kw_range = self.matcher.search(kw.replace('  ', ' '))

            if kw_range is None:
                return [], []

            tok_start, tok_end = kw_range.start(), kw_range.end()
        else:
            tok_start, tok_end = tok_range

        tmp = self.matcher.pos_index[tok_start: tok_end + 1]
        start, end = tmp[0], tmp[-1]
        shot_text = self.matcher.text[start:end]

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

    def _load_item(self, item):
        result, convert_result = self._init_polar_result(item)
        result_item, convert_result_item = self._gen_result(item)
        convert_result.extend(convert_result_item)
        result.extend(result_item)

        return result, convert_result

    def _fix_term(self, item):
        m = TokenMatcher(item)
        if len(m.tokens) > 10:
            return ''.join(m.tokens[:5]) + ' ' + ''.join(m.tokens[-5:])
        return item

    def _preprocess_item(self, item):
        # note: 当opinion_column字段值比较长, token 大于11, ainlp 引用的 pyrefo.findall 函数会报错, 暂时不知道原因
        # 临时解决办法是 在预处理的时候进行检查如果发现比较长, 人工截断 (csk 20210601)

        new_item = {}
        aspect_term, opinion_term = item[self.aspect_column], item[self.opinion_column]
        if self.lang == 'cn':
            opinion_term = self._fix_term(opinion_term)
            aspect_term = self._fix_term(aspect_term)

        new_item[self.aspect_column] = aspect_term.strip().split(',') if isinstance(aspect_term, str) and len(
            aspect_term) > 0 else []
        new_item[self.opinion_column] = opinion_term.strip().split(',') if isinstance(opinion_term, str) and len(
            opinion_term) > 0 else []

        new_item['raw_text'] = item[self.text_column]
        new_item[self.text_column] = self.processor.process(item[self.text_column].lower())
        new_item['nlp_level'] = '#'.join([item[level] for level in self.level_columns])

        for column in [self.choice_column] + self.keep_columns:
            new_item[column] = item[column]
        return new_item

    def init_matcher(self, text):
        del self.matcher
        self.matcher = TokenMatcher(text, clause_delimiter=self.clause_delimiter)

    def load(self, item):
        """convert line of nlp csv to a label-studio absa task"""
        item = self._preprocess_item(item)
        self.init_matcher(item[self.text_column])

        result, convert_result = self._load_item(item)
        if len(result) == 1:
            # 重新查找
            try:
                has_error, new_item, reason = self._check_item(item, self.matcher.tokens)
                if has_error and new_item is not None:
                    result, convert_result = self._load_item(new_item)
            except Exception as e:
                logger.error(repr(e))

        if len(result) == 1:
            return False, {}

        else:
            return True, {
                'data': self._prepare_data(item),
                'completions': [
                    {
                        'result': result,
                        'convert_result': convert_result,
                    }
                ],
            }

    def _prepare_data(self, item):
        data = {
            'text': ' '.join(self.matcher.tokens) if self.lang == "cn" else item[self.text_column],
            'original_text': item[self.text_column],
            'raw_text': item['raw_text'],
            'nlp_level': item['nlp_level']
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
