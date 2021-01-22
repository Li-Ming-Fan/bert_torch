

import json
import copy

#
dict_config_bert_base_chinese = {
    "attention_probs_dropout_prob": 0.1,
    "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "type_vocab_size": 2,
    "vocab_size": 21128
}


"""
dict_config_bert_base_chinese = {
    "attention_probs_dropout_prob": 0.1,
    "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "pooler_fc_size": 768,
    "pooler_num_attention_heads": 12,
    "pooler_num_fc_layers": 3,
    "pooler_size_per_head": 128,
    "pooler_type": "first_token_transform",
    "type_vocab_size": 2,
    "vocab_size": 21128
}

"""


class BertConfig(object):
    """
    """
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
        gradient_checkpointing=False,
        is_decoder=False,
        add_cross_attention=False,
        **kwargs ):
        """
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.return_dict = return_dict
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.gradient_checkpointing = gradient_checkpointing
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention

    def update(self, config_dict):
        """
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    #
    def from_dict(self, config_dict):
        """
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def from_json_file(self, json_file):
        """
        """
        with open(json_file, "r", encoding="utf-8") as fp:
            settings = json.load(fp)
        #
        self.from_dict(settings)
        #

    #
    def to_dict(self):
        """
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self):
        """
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=4, sort_keys=True)

    def to_json_file(self, json_file_path):
        """
        """
        with open(json_file_path, "w", encoding="utf-8") as fp:
            fp.write(self.to_json_string())
        #
