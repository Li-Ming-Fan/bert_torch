
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from metrics_np import Metrics

from .prelm.bert_modeling import BertModel


#
class ModelClassifier(nn.Module):
    """
    """
    def __init__(self, settings):
        """
        """
        super(ModelClassifier, self).__init__()
        #
        self.settings = settings
        #
        self.bert = BertModel(settings.bert_config, add_pooling_layer=True)
        self.dropout = nn.Dropout(settings.bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(settings.bert_config.hidden_size, settings.num_labels)
        #
        self.loss_function = nn.CrossEntropyLoss()
        #
    #
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None,
                      is_training=True, return_dict=False):
        """
        """
        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            return_dict=False)
        #
        pooled_output = outputs[0]
        # sequence_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        #
        if is_training:
            loss_train = self.get_loss(logits, labels)
            return loss_train, logits
        else:
            return (logits, )
        #
    #
    def get_loss(self, logits, labels):
        """
        """
        return self.loss_function(logits, labels)
        #
    #
#

#
def convert_single_example_tokens(tokenizer, tokens_a, tokens_b=None, label_id=None,
                        max_seq_length=128, logger=None):
    """
    """
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    d = max_seq_length - len(input_ids)
    if d > 0:
        input_ids.extend( [0] *d )
        input_mask.extend( [0] *d )
        segment_ids.extend( [0] *d )
    #
    # while len(input_ids) < max_seq_length:
    #     input_ids.append(0)
    #     input_mask.append(0)
    #     segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    #
    return (input_ids, input_mask, segment_ids, label_id)
    #
#
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
#
def convert_list_examples_to_batch(list_examples, list_keys=None):
    """
    """
    batch_tuple = list(zip(*list_examples))
    #    
    if list_keys:
        #
        batch_dict = {}
        for idx, key in enumerate(list_keys):
            batch_dict[key] = batch_tuple[idx]
        #
        return batch_dict
        #
    else:
        return batch_tuple
        #
    #
#
def dataset_creator(tokenizer, examples, label_list, max_seq_len=128, logger=None):
    """
    """
    labels_map = {}
    for idx, item in enumerate(label_list):
        labels_map[item] = idx
    #
    # if output_mode == "classification":
    #     output_type = torch.long
    # elif output_mode == "regression":
    #     output_type = torch.float
    # #

    #
    features = []
    for idx, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        #
        label_id = labels_map[example.label]
        #
        feature = convert_single_example_tokens(
            tokenizer, tokens_a, tokens_b, label_id,
            max_seq_length=max_seq_len, logger=logger)
        #  (input_ids, input_mask, segment_ids, label_id)
        #
        features.append(feature)
        #
        # logger
        if idx < 5:
            logger.info("example: %d" % idx)
            logger.info("tokens_a: %s" % (" ".join(tokens_a)))
            if tokens_b: logger.info("tokens_b: %s" % (" ".join(tokens_b)))
            logger.info("label: %s" % example.label)
            #
            logger.info("input_ids: %s" % (" ".join([str(tid) for tid in feature[0]])))
            logger.info("input_mask: %s" % (" ".join([str(tid) for tid in feature[1]])))
            logger.info("segment_ids: %s" % (" ".join([str(tid) for tid in feature[2]])))
            logger.info("example end.")
            #
        #
    #

    #
    # dataset
    #
    single_batch = list(zip(*features))
    #
    list_tensors = []
    #
    for idx in range(len(single_batch) ):
        list_tensors.append(
            torch.tensor(single_batch[idx], dtype=torch.long) )
    #
    dataset = TensorDataset(*list_tensors)
    #
    return dataset
    #
#

#
def scores_calculator(list_batches_evaluated, is_training=False):
    """ list_batches_evaluated: list of (batch_input_tuple, model_output)
    """
    scores_dict = {}
    #
    list_labels = []
    list_logits = []
    #
    for batch, output in list_batches_evaluated:
        labels_curr = batch[-1]   #
        logits_curr = output[0]   #
        #
        list_labels.extend(labels_curr.tolist())
        list_logits.extend(logits_curr.tolist())
        #
    #
    scores_dict = Metrics.classification_scores(list_logits, list_labels)
    print("scores: {}".format(scores_dict))
    #
    main_score = scores_dict["macro avg"]["f1-score"]
    print("main_score: {}".format(main_score))
    #
    return main_score, scores_dict
    #
#



