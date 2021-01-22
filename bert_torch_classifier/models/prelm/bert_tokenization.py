
import os
import copy
import collections

import unicodedata

#
def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


#
def write_vocab_file(vocab_file, index2token):
    """
    """
    fp = open(vocab_file, "w", encoding="utf-8")
    for index, token in index2token.items():
        fp.write("%s\n" % token)
    fp.close()
    #
#
def load_vocab_file(vocab_file):
    """
    """
    # token2index = collections.OrderedDict()
    index2token = collections.OrderedDict()
    # index2token = {}
    #
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        index2token[index] = token
    #
    return index2token
    #
#
def whitespace_tokenize(text):
    """
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens
#
class BertTokenizer(object):
    """
    """
    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True,
        never_split=[],
        customized_id_start=1,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        write_temp_file=True, **kwargs):
        """
        """
        self.vocab_file=vocab_file
        self.do_lower_case=do_lower_case
        self.do_basic_tokenize=do_basic_tokenize
        self.never_split=never_split
        self.customized_id_start=customized_id_start
        self.pad_token=pad_token
        self.unk_token=unk_token
        self.cls_token=cls_token
        self.sep_token=sep_token
        self.mask_token=mask_token
        self.tokenize_chinese_chars=tokenize_chinese_chars
        self.strip_accents=strip_accents
        self.write_temp_file=write_temp_file
        #
        # index2token, load
        if not os.path.isfile(self.vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(self.vocab_file)
            )
        #
        self.index2token = load_vocab_file(self.vocab_file)   # index2token
        # self.vocab = load_vocab_file(self.vocab_file)   # token2index
        #

        #
        # special_tokens
        self.all_special_tokens = [
            self.pad_token, self.unk_token, self.cls_token, self.sep_token,
            self.mask_token
        ]
        #
        # never_split
        self.never_split_extended = copy.deepcopy(self.never_split)
        #
        for item in self.all_special_tokens:
            self.never_split_extended.append(item)
        #
        # modified_vocab
        customized_id = self.customized_id_start
        for item in self.never_split:
            self.index2token[customized_id] = item
            customized_id += 1
            #
        #

        #
        # token2index, vocab
        self.vocab = collections.OrderedDict(
            [(tok, idx) for idx, tok in self.index2token.items()])
        #
        self.pad_id = self.vocab[self.pad_token]
        self.unk_id = self.vocab[self.unk_token]
        #
        self.cls_id = self.vocab[self.cls_token]
        self.sep_id = self.vocab[self.sep_token]
        #

        #
        if self.write_temp_file:
            file_path = "zzz_temp_vocab_file.txt"
            write_vocab_file(file_path, self.index2token)
        #
        
        #
        # tokenizer
        if self.do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=self.do_lower_case,
                never_split=self.never_split_extended,
                tokenize_chinese_chars=self.tokenize_chinese_chars,
                strip_accents=self.strip_accents )
            #
        #
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=self.unk_token)
        #
    #
    @property
    def vocab_size(self):
        return len(self.vocab)
    #
    def get_vocab(self):
        return dict(self.vocab)
    #

    #
    def tokenize(self, text):
        """
        """
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens
    #
    def tokenize_and_convert_to_idx(self, text):
        """
        """
        return [self.vocab.get(w, self.unk_id) for w in self.tokenize(text)]
        #
    #

    #
    def convert_ids_to_tokens(self, list_idx):
        """
        """
        return [self.index2token.get(tid, self.unk_token) for tid in list_idx]
        #
    #
    def convert_tokens_to_ids(self, tokens):
        """
        """
        return [self.vocab.get(w, self.unk_id) for w in tokens]
        #
    #
    def convert_tokens_to_string(self, tokens):
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string
    #

    #
    def convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_id)
    #
    def convert_id_to_token(self, index):
        return self.index2token.get(index, self.unk_token)
    #
    
    #
    def build_bert_inputs(self, tokens_id_a, tokens_id_b=None, max_seq_len=256):
        """
        """
        if tokens_id_b is None:
            input_seq = [self.cls_id] + tokens_id_a + [self.sep_id]
            segment_seq = [0] * (len(tokens_id_a) + 2)
        else:
            cls_id = [self.cls_id]
            sep_id = [self.sep_id]
            #
            input_seq = cls_id + tokens_id_a + sep_id + tokens_id_b + sep_id
            segment_seq = [0] * (len(tokens_id_a) + 2) + [1] * (len(tokens_id_b) + 1)
            #
        #
        mask_seq = [1] * len(input_seq)
        #
        # [PAD]
        #
        d = max_seq_len - len(input_seq)
        #
        if d > 0:
            input_seq = input_seq + [self.pad_id] * d
            segment_seq = segment_seq + [self.pad_id] * d
            mask_seq = mask_seq + [0] * d
        elif d < 0:
            str_info = "max_seq_len < len(input_seq), do truncate tokens_id_a and/or tokens_id_b"
            assert False, str_info
        #
        return input_seq, mask_seq, segment_seq
        #
    #
    def build_sequence_a_mask(self, tokens_id_a, tokens_id_b=None, max_seq_len=256):
        """
        """
        len_a = len(tokens_id_a)
        #
        return [0] + [1] * len_a + [0] * (max_seq_len - len_a - 1)
        #
    #
    def build_span_mask(self, posi_s, span_len, max_seq_len=256):
        """
        """
        return [0] * posi_s + [1] * span_len + [0] * (max_seq_len - posi_s - span_len)
        #
    #    
#

#
class BasicTokenizer(object):
    """
    """
    def __init__(self, do_lower_case=True, never_split=[],
            tokenize_chinese_chars=True, strip_accents=None):
        """
        """
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        """
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)
        #
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        #
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """ Performs invalid character removal and whitespace cleanup on text. 
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
            #
        #
        return "".join(output)
        #
    #
#
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
#


