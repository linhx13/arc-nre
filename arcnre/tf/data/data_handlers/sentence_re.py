from typing import Dict, Iterable
import json
from collections import Counter

import tensorflow as tf

from .data_handler import DataHandler
from ..fields import Field
from ...vocabs import Vocab


class SentenceREDataHandler(DataHandler):

    def __init__(self, text_field, pos_field, label_field):
        self.text_field = text_field
        self.pos_field = pos_field
        self.label_field = label_field
        super(SentenceREDataHandler, self).__init__(
            {'token': text_field,
             'pos_head': self.pos_field, 'pos_tail': self.pos_field},
            {'relation': label_field}
        )

    def read_examples(self, path) -> Iterable[Dict]:
        with open(path) as fin:
            for line in fin:
                data = json.loads(line)
                yield data

    def encode_example(self, data: Dict) -> Dict:
        example = {}
        example['token'] = self.text_field.encode(data['token'])
        example['pos_head'] = self.pos_field.encode(data['token'], data['h'])
        example['pos_tail'] = self.pos_field.encode(data['token'], data['t'])
        if data.get("relation") is not None:
            example['relation'] = self.label_field.encode(data['relation'])
        return example

    def build_vocab(self, *args, **kwargs):
        text_counter, label_counter = Counter(), Counter()
        for examples in args:
            for ex in examples:
                self.text_field.count_vocab(ex['token'], text_counter)
                self.label_field.count_vocab(ex['relation'], label_counter)
        self.text_field.vocab = Vocab(text_counter)
        self.label_field.vocab = Vocab(label_counter, unknown_token=None,
                                       reserved_tokens=[])

    def element_length_func(self, example) -> int:
        return tf.shape(example['token'])[0]
