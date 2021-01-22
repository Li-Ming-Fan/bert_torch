
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time
import logging
import collections

import csv
import pandas as pd


#
class InputExample(object):
    """
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
#
class InputFeatures(object):
    """
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        """
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
#
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
    #
#
class SimProcessor(DataProcessor):
    """
    """
    #
    def get_examples_from_lines(self, lines, data_tag):
        """
        """
        data_loaded = []
        #
        index = 0
        for line in lines:
            guid = '%s-%d' % (data_tag, index)
            line = line.replace("\n", "").split("\t")
            #
            # text_a = tokenization.convert_to_unicode(str(line[1]))
            text_a = str(line[1])
            label = str(line[2])
            #
            data_loaded.append(InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label))
            #
            index += 1
            #
        #
        return data_loaded
        #
    #
    def get_train_examples(self, data_dir):
        """
        """
        data_tag = "train"
        file_path = os.path.join(data_dir, 'train_sentiment.txt')
        #
        fp = open(file_path, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()
        #
        return self.get_examples_from_lines(lines, data_tag)
        #
    #
    def get_dev_examples(self, data_dir):
        """
        """
        data_tag = "valid"
        file_path = os.path.join(data_dir, 'test_sentiment.txt')
        #
        fp = open(file_path, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()
        #
        return self.get_examples_from_lines(lines, data_tag)
        #
    #
    def get_test_examples(self, data_dir):
        """
        """
        data_tag = "test"
        file_path = os.path.join(data_dir, 'test_sentiment.txt')
        #
        fp = open(file_path, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()
        #
        return self.get_examples_from_lines(lines, data_tag)
        #
    #
    def get_labels(self):
        return ['0', '1', '2']
    #
    def get_class_weights(self):
        return [1, 1, 3]
    #
#

