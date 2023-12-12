import random
from collections import Counter
from typing import (
    List,
    Iterable,
    Set,
    Union,
    Dict,
    overload,
    Sequence,
    Mapping,
    Literal,
    Tuple,
    Optional,
)

import jsonlines
import networkx as nx
import numpy as np
import spacy
import torch
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    XLNetTokenizer,
    AlbertTokenizer,
)
from math import ceil

from NewsSentiment.SentimentClasses import SentimentClasses
from NewsSentiment.customexceptions import TooLongTextException, TargetNotFoundException
from NewsSentiment.consts import *
from NewsSentiment.fxlogger import get_logger

# get logger
from NewsSentiment.knowledge.bingliuopinion.bingliuopinion import (
    get_bingliu_polarities_as_tensor,
    get_num_bingliu_polarities,
)
from NewsSentiment.knowledge.mpqasubjectivity.mpqasubjectivity import (
    get_mpqa_subjectivity_polarities_as_tensor,
    get_num_mpqa_subjectivity_polarities,
)
from NewsSentiment.knowledge.nrcemolex.nrcemolex import (
    get_nrc_emotions_as_tensor,
    get_num_nrc_emotions,
)
from NewsSentiment.knowledge.zeros.zerosknowledge import (
    get_num_zero_dimensions,
    get_zeros_as_tensor,
)
from NewsSentiment.models.FXBaseModel import FXBaseModel

logger = get_logger()


class RandomOversampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset: Dataset, random_seed=None):
        x = []
        y = []
        for ind, example in enumerate(dataset):
            x.append(ind)
            y.append(example["polarity"])

        x_arr = np.asarray(x).reshape((len(x), 1))
        y_arr = np.asarray(y).ravel()

        ros = RandomOverSampler(random_state=random_seed)
        x_sampled, y_sampled = ros.fit_resample(x_arr, y_arr)
        self.sampled_indexes = x_sampled.ravel().tolist()
        sampled_labels = y_sampled.tolist()

        assert len(self.sampled_indexes) == len(sampled_labels)

        random.shuffle(self.sampled_indexes)

        get_logger().info(
            f"oversampled to {len(self.sampled_indexes)} samples. label distribution: "
            f"{Counter(sampled_labels)}"
        )

    def __len__(self):
        return len(self.sampled_indexes)

    def __iter__(self):
        return iter(self.sampled_indexes)


class FXEasyTokenizer:
    NLP = None
    NLP_DEP_PARSER_LABELS = None
    NUM_CATEGORIES_OF_SELECTED_KNOWLEDGE_SOURCES = 0
    __PROCESSED_KNOWLEDGE_SOURCES = set()

    def __init__(
        self,
        tokenizers_name_and_obj: dict,
        max_seq_len: int,
        knowledge_sources: Iterable[str],
        is_use_natural_target_phrase_for_spc: bool,
    ):
        self._get_labels()
        self.tokenizers_name_and_obj = tokenizers_name_and_obj
        self.max_seq_len = max_seq_len
        self.knowledge_sources = knowledge_sources
        self.is_use_natural_target_phrase_for_spc = is_use_natural_target_phrase_for_spc

    @classmethod
    def _get_labels(cls):
        if cls.NLP_DEP_PARSER_LABELS is not None:
            return
        try:
            cls.NLP = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            cls.NLP = spacy.load("en_core_web_sm")

        # get list of parser's labels
        parser_index = cls.NLP.pipe_names.index("parser")
        cls.NLP_DEP_PARSER_LABELS = list(cls.NLP.pipeline[parser_index][1].labels)

    @staticmethod
    @overload
    def create_entire_text(
        text_left: str,
        target_phrase: str,
        text_right: str,
        is_return_modified_text_left: Literal[True],
    ) -> Tuple[str, str]:
        ...

    @staticmethod
    @overload
    def create_entire_text(
        text_left: str,
        target_phrase: str,
        text_right: str,
        is_return_modified_text_left: Literal[False],
    ) -> str:
        ...

    @staticmethod
    def create_entire_text(
        text_left: str,
        target_phrase: str,
        text_right: str,
        is_return_modified_text_left: bool,
    ) -> Union[Tuple[str, str], str]:
        """
        Create the entire text from the three prepared text segment. Will modify
        text_left (append a whitespace " ") if it is empty, which is necessary for
        target mask creation.
        :param is_return_modified_text_left:
        :param text_left:
        :param target_phrase:
        :param text_right:
        :return:
        """
        # input text must have basic tokenization, i.e., each token is separated by ' '
        if len(text_left) > 0:
            assert text_left[-1] == " ", f"left is not prepared: '{text_left[-1]}'"
        else:
            # we need to have a non-empty left sequence, otherwise BERT's tokenizer
            # will raise an exception
            text_left = " "
        if len(text_right) > 0:
            assert text_right[0] == " ", f"right is not prepared: '{text_right[0]}'"

        # produce entire text string
        text = text_left + target_phrase + text_right

        if is_return_modified_text_left:
            return text, text_left
        else:
            return text

    @staticmethod
    def prepare_left_segment(text_left: str):
        """
        Prepares the left text segment. See FXDataset:task_to_dataset_item for more
        information. Removes any leading newlines.
        :param text_left:
        :return:
        """
        text_left = text_left.lstrip("\n")
        if len(text_left) > 0 and text_left[-1] != " ":
            text_left += " "
        return text_left

    @staticmethod
    def prepare_target_mention(target_mention: str):
        """
        Removes any leading newlines.
        :param target_mention:
        :return:
        """
        return target_mention.lstrip("\n")

    @staticmethod
    def prepare_right_segment(text_right: str):
        """
        Prepares the right text segment. See FXDataset:task_to_dataset_item for more
        information.
        :param text_right:
        :return:
        """
        if len(text_right) > 0 and text_right[0] != " ":
            text_right = " " + text_right
        return text_right

    def _create_word_to_wordpiece_mapping(
        self, tokenizer, words: List, for_text_with_special_tokens, offset
    ):
        batch_output = self._batch_create_word_to_wordpiece_mapping(
            tokenizer, [words], for_text_with_special_tokens, [offset]
        )
        return tuple(output[0] for output in batch_output)

    def _batch_create_word_to_wordpiece_mapping(
        self,
        tokenizer,
        words_per_target: List[List],
        for_text_with_special_tokens,
        offsets,
    ):
        if not for_text_with_special_tokens:
            raise NotImplementedError()

        target_words = [
            (
                " ".join(target[:word_index]) + " ",  # left
                word,  # target
                "",  # right
            )
            for target in words_per_target
            for word_index, word in enumerate(target)
        ]

        # produce target masks
        target_masks = self._batch_create_target_mask(
            tokenizer=tokenizer,
            targets=target_words,
            for_text_with_special_tokens=for_text_with_special_tokens,
            is_raise_exception_if_target_after_max_seq_len=False,
        )

        # split back into targets
        _indexer = 0
        target_masks_per_target = [
            target_masks[_indexer : (_indexer := _indexer + len(wpt))]
            for wpt in words_per_target
        ]

        # remove "toolong" (word is after the max seq len)
        valid_target_masks_per_target = []
        valid_tokens = []
        for target, tokens in zip(target_masks_per_target, words_per_target):
            indexer = target.index("toolong") if "toolong" in target else None
            valid_target_masks_per_target.append(target[:indexer])
            valid_tokens.append(tokens[:indexer])

        # convert to tensors
        mappings = []

        for target, offset in zip(valid_target_masks_per_target, offsets):
            target_tensor = torch.LongTensor(target)

            word_indices_multiplier = torch.tensor(
                range(offset, offset + len(target_tensor))
            )
            target_tensor = target_tensor.mul(word_indices_multiplier.unsqueeze(-1))

            if type(tokenizer) == RobertaTokenizer:
                if target_tensor.prod(0).sum(0) > 0:
                    logger.debug(
                        "overlap when mapping tokens to wordpiece (allow overwriting because"
                        " Roberta is used)"
                    )
            else:
                assert target_tensor.prod(0).sum(0) == 0

            target_tensor = target_tensor.sum(0)

            assert target_tensor.size(0) <= self.max_seq_len

            mappings.append(target_tensor)

        return mappings, valid_tokens

    def _calculate_dep_matrix(self, text_tokens, text_tokens_as_str):
        _check_len_doc = len(text_tokens)
        assert _check_len_doc == len(
            text_tokens_as_str
        ), f"{_check_len_doc} vs {len(text_tokens_as_str)}: {text_tokens} vs {text_tokens_as_str}"

        # for the dependency matrix
        dependency_tensor = torch.zeros(
            len(text_tokens_as_str), len(text_tokens_as_str), dtype=torch.long
        )

        for token_index, token in enumerate(text_tokens):
            if token.is_space:
                # text_tokens_as_str does not contain whitespace tokens, so we skip them
                # here, too
                continue
            assert (
                token.text == text_tokens_as_str[token_index]
            ), f"'{token.text}' vs. '{text_tokens_as_str[token_index]}'"

            # to build the dependency relation matrix
            # get the token's head and type of relation
            head_of_token = token.head
            try:
                index_in_token_list_of_head_of_token = text_tokens.index(head_of_token)
            except ValueError as ve:
                # if the head cannot be found in the (truncated to wordpiece equivalent
                # max seq len) list of text_tokens, this means that the current token's
                # head is not part of the wordpiece sequence. we print an error and
                # not change the current word's vector (will be all 0's)
                logger.warning(ve)
                continue

            relation_to_head = token.dep_
            # offset the relation by 1 so that the root relation (which is 0) is
            # non-zero
            index_of_relation_to_head = (
                self.NLP_DEP_PARSER_LABELS.index(relation_to_head) + 1
            )
            # insert to dependency tensor
            dependency_tensor[
                token_index, index_in_token_list_of_head_of_token
            ] = index_of_relation_to_head

        return dependency_tensor

    def _calculate_dep_distance(self, text_tokens, text_left_len, target):
        threshold_for_target_token = 8

        # by using spacys tokenization also on the target (instead of a simple
        # whitespace split as in https://github.com/StevePhan101/LCFS-BERT/
        # we ensure that the same tokenization as was used for the text is applied for
        # the target
        nlp_target = self.NLP(target)
        # target_terms_lowercased = [a.lower() for a in target.split()]
        target_terms_lowercased = [a.text.lower() for a in nlp_target]

        # Load spacy's dependency tree into a networkx graph
        edges = []
        cnt = 0
        found_start_of_target = False
        target_term_ids = [0] * len(target_terms_lowercased)
        len_previously_processed_tokens = 0
        for token in text_tokens:
            # Record the position of aspect terms
            lowercased_token = token.lower_
            token_startchar_indoc = token.idx
            token_endchar_indoc = token_startchar_indoc + len(token.text_with_ws)

            if (
                cnt < len(target_terms_lowercased)
                and lowercased_token == target_terms_lowercased[cnt]
            ):
                # if both conditions above are satisfied, we probably have (part of)
                # the target in the current token. however, to be slightly more
                # confident about this, also ensure that the previous tokens have
                # approximately the same length as text_left_len. or skip that check
                # if we already found the start of the target
                if (
                    found_start_of_target
                    or abs(text_left_len - len_previously_processed_tokens)
                    < threshold_for_target_token
                ):
                    found_start_of_target = True
                    target_term_ids[cnt] = token.i
                    cnt += 1
                else:
                    pass

            len_previously_processed_tokens += len(token.text_with_ws)

            # for lcfs absa
            for child in token.children:
                edges.append(
                    (
                        "{}_{}".format(token.lower_, token.i),
                        "{}_{}".format(child.lower_, child.i),
                    )
                )

        if not found_start_of_target:
            raise TargetNotFoundException(
                f"no target found: {text_tokens}, {text_left_len}, {target}"
            )

        # create the graph
        graph = nx.Graph(edges)

        dist = [0.0] * len(text_tokens)
        for i, word in enumerate(text_tokens):
            source = "{}_{}".format(word.lower_, word.i)
            sum = 0
            for term_id, term in zip(target_term_ids, target_terms_lowercased):
                target = "{}_{}".format(term, term_id)
                try:
                    sum += nx.shortest_path_length(graph, source=source, target=target)
                except:
                    sum += len(text_tokens)  # No connection between source and target
            dist[i] = sum / len(target_terms_lowercased)
        return dist

    def _create_target_mask(
        self,
        tokenizer,
        text_left: str,
        target: str,
        text_right: str,
        for_text_with_special_tokens: bool,
        is_raise_exception_if_target_after_max_seq_len: bool = True,
    ):
        """
        Creates a target mask for a single target in a text that comprises left, target,
        and right. Specifically, the target mask has the same length as the sequence of
        the entire text including special tokens, and contains 1's for each position k
        where in the entire text sequence (including special tokens) the token at
        position k is part of the target phrase.
        Simplified example (this does not consider any LM specific tokenization
        characteristics, e.g., most LMs' tokenizers do not tokenize single tokens but
        produce sub-tokens (e.g., BERT using the WordPiece tokenizer).
        Text:                                    Mr. <target>Smith</target> is great .
        Text indexes (with special tokens):  CLS 12           50            18 15    19 END
        Target mask:                          0  0             1             0  0    0   0
        :param tokenizer:
        :param text_left:
        :param target:
        :param text_right:
        :param text_pair: if not None, will be appended to the input sequence as the second pair
        :return:
        """
        return self._batch_create_target_mask(
            tokenizer,
            ((text_left, target, text_right),),
            for_text_with_special_tokens,
            is_raise_exception_if_target_after_max_seq_len,
        )[0]

    def _batch_create_target_mask(
        self,
        tokenizer,
        targets,
        for_text_with_special_tokens: bool,
        is_raise_exception_if_target_after_max_seq_len: bool = True,
    ):
        if not for_text_with_special_tokens:
            raise NotImplementedError()

        lefts = []
        rights = []
        full_texts = []
        target_phrases = []
        for target in targets:
            ft, l = self.create_entire_text(*target, is_return_modified_text_left=True)
            lefts.append(l)
            full_texts.append(ft)
            target_phrases.append(target[1])
            rights.append(target[2])

        encodings = self._encode_for_target_mask(
            tokenizer, lefts, target_phrases, rights, full_texts
        )

        return [
            self._create_target_mask_on_encoding(
                tokenizer, *encoding, is_raise_exception_if_target_after_max_seq_len
            )
            for encoding in zip(*encodings)
        ]

    def _encode_for_target_mask(
        self, tokenizer, text_lefts, targets, text_rights, full_texts
    ):
        # get token ids, cf. https://huggingface.co/transformers/glossary.html
        text_left_ids_with_special_tokens = tokenizer(
            text_lefts, add_special_tokens=True
        )["input_ids"]
        text_right_ids = tokenizer(text_rights, add_special_tokens=False)["input_ids"]
        text_ids_with_special_tokens = tokenizer(full_texts, add_special_tokens=True)[
            "input_ids"
        ]
        target_phrase_ids_with_special_tokens = tokenizer(
            targets, add_special_tokens=True
        )["input_ids"]

        return (
            text_left_ids_with_special_tokens,
            target_phrase_ids_with_special_tokens,
            text_right_ids,
            text_ids_with_special_tokens,
        )

    def _create_target_mask_on_encoding(
        self,
        tokenizer,
        text_left_ids_with_special_tokens,
        target_phrase_ids_with_special_tokens,
        text_right_ids,
        text_ids_with_special_tokens,
        is_raise_exception_if_target_after_max_seq_len,
    ):
        len_text_left_ids_with_special_tokens = len(text_left_ids_with_special_tokens)
        len_text_right_ids = len(text_right_ids)
        len_text_id_with_special_tokens = len(text_ids_with_special_tokens)
        len_target_ids_with_special_tokens = len(target_phrase_ids_with_special_tokens)

        token_obj_class = type(tokenizer)
        if token_obj_class in [BertTokenizer, RobertaTokenizer, AlbertTokenizer]:
            # these tokenizers create 1 special token at the end that should not be
            num_special_tokens_at_start = 1
            num_special_tokens_at_end = 1
        elif token_obj_class == XLNetTokenizer:
            num_special_tokens_at_start = 0
            num_special_tokens_at_end = 2
        else:
            raise ValueError
        # determine numbers of special tokens in the target
        num_special_tokens_in_target = (
            num_special_tokens_at_start + num_special_tokens_at_end
        )
        # get the number of token ids of the target, excluding special tokens
        # we do this, because
        num_effective_length_target_in_text = (
            len_target_ids_with_special_tokens - num_special_tokens_in_target
        )
        # get the position of the first non-special token of the target phrase in the
        # sequence of tokens representing the entire text, which includes special tokens
        tp_seq_in_text_seq_pos_start = (
            len_text_left_ids_with_special_tokens - num_special_tokens_at_end
        )
        # likewise, get the position of the last non-special token of the target phrase
        # in the sequences of tokens representing the entire text, which includes
        # special tokens
        tp_seq_in_text_seq_pos_end = (
            tp_seq_in_text_seq_pos_start + num_effective_length_target_in_text
        )
        # get the length of the token sequence representing the target phrase (again,
        # only actual content tokens, no special tokens)
        tp_seq_in_text_seq_len = (
            tp_seq_in_text_seq_pos_end - tp_seq_in_text_seq_pos_start
        )
        assert (
            tp_seq_in_text_seq_len > 0
        ), f"tp_seq_in_text_seq_len: {tp_seq_in_text_seq_len}"

        # if the end position of the target is greater than the maximum sequence length,
        # we raise an Exception, since it won't make sense to train with or evaluate on
        # this example since the target will be out of bounds
        if tp_seq_in_text_seq_pos_end > self.max_seq_len:
            if is_raise_exception_if_target_after_max_seq_len:
                raise TooLongTextException()
            else:
                # e.g., if invoked for creating the target mapping
                return "toolong"

        # create target mask: all 0 except for occurrence of target
        target_mask_seq = [0] * self.max_seq_len
        target_mask_seq[tp_seq_in_text_seq_pos_start:tp_seq_in_text_seq_pos_end] = [
            1
        ] * tp_seq_in_text_seq_len

        # perform checks
        count_nonzero_elements_in_target_mask_seq = sum(
            [1 if elem != 0 else 0 for elem in target_mask_seq]
        )

        # roberta: 1 special token at beginning, 1 at end
        # bert: 1 beginning, 1 at end
        # xlnet: 0 beginning, 2 at end
        # len_text_left_ids_with_special_tokens contains the right amount of both
        # special tokens at the beginning and end of the sequence
        calculated_len = (
            len_text_left_ids_with_special_tokens
            + num_effective_length_target_in_text
            + len_text_right_ids
        )

        if token_obj_class == RobertaTokenizer:
            # Roberta has more complex tokenization behavior, e.g., >>> tokenizer("the friendly staff welcomes")['input_ids']
            # [0, 627, 5192, 813, 16795, 2]
            # >>> tokenizer("welcomes")['input_ids']
            # [0, 605, 523, 32696, 2]
            # >>> tokenizer("welcomes you")['input_ids']
            # [0, 605, 523, 32696, 47, 2]
            # -> so we skip this check here
            pass
        else:
            assert (
                len_text_id_with_special_tokens == calculated_len
            ), f"{len_text_id_with_special_tokens} == {calculated_len}"
        assert (
            count_nonzero_elements_in_target_mask_seq
            == num_effective_length_target_in_text
        )

        return target_mask_seq

    def _create_knowledge_source_tensor(
        self, text_tokens_as_str: List, offset: int, mapping, mode: str
    ):
        if mode == "nrc_emotions":
            num_categories = get_num_nrc_emotions()
        elif mode == "mpqa_subjectivity":
            num_categories = get_num_mpqa_subjectivity_polarities()
        elif mode == "bingliu_opinion":
            num_categories = get_num_bingliu_polarities()
        elif mode == "liwc":
            from knowledge.liwc.liwc import get_num_liwc_categories

            num_categories = get_num_liwc_categories()
        elif mode == "zeros":
            num_categories = get_num_zero_dimensions()
        else:
            raise NotImplementedError

        emotions_for_sequence = torch.zeros(
            self.max_seq_len, num_categories, dtype=torch.long
        )
        for word_index, word in enumerate(text_tokens_as_str, start=offset):
            positions_of_current_word_in_input_sequence = (
                self._map_token_index_to_wordpiece_index(mapping, word_index)
            )
            if mode == "nrc_emotions":
                emotion_tensor = get_nrc_emotions_as_tensor(word)
            elif mode == "mpqa_subjectivity":
                emotion_tensor = get_mpqa_subjectivity_polarities_as_tensor(word)
            elif mode == "bingliu_opinion":
                emotion_tensor = get_bingliu_polarities_as_tensor(word)
            elif mode == "liwc":
                from knowledge.liwc.liwc import get_liwc_categories_as_tensor

                emotion_tensor = get_liwc_categories_as_tensor(word)
            elif mode == "zeros":
                emotion_tensor = get_zeros_as_tensor(word)
            else:
                raise NotImplementedError

            for position_in_seq in positions_of_current_word_in_input_sequence:
                emotions_for_sequence[position_in_seq, :] = emotion_tensor
        return emotions_for_sequence

    def _create_mapping_from_tokenbased_to_wordpiece_based(self, text, tok_obj):
        batch_output = self._batch_create_mapping_from_tokenbased_to_wordpiece_based(
            (text,), tok_obj
        )
        return tuple(output[0] for output in batch_output)

    def _batch_create_mapping_from_tokenbased_to_wordpiece_based(self, texts, tok_obj):
        offset = 10
        # for spacy, we need to remove leading spaces as they will yield a single
        # token
        text_without_leading_space = [text.strip() for text in texts]
        nlp_text_per_target = self.NLP.pipe(text_without_leading_space)

        # get only non-single-space tokens, see
        # https://github.com/explosion/spaCy/issues/1707
        text_tokens, text_tokens_as_str = zip(
            *(
                zip(*((token, token.text) for token in nlp_text if not token.is_space))
                for nlp_text in nlp_text_per_target
            )
        )
        # in case there is a word at position k that is longer than the max seq len
        # when converted to wordpiece indexes, _create_word_to_wordpiece_mapping
        # returns all words up to including k-1
        mapping, text_tokens_as_str = self._batch_create_word_to_wordpiece_mapping(
            tok_obj,
            text_tokens_as_str,
            for_text_with_special_tokens=True,
            offsets=[offset] * len(texts),
        )
        # ..., correspondingly, truncate text_tokens to same length
        text_tokens = [
            tt[: len(ttas)] for tt, ttas in zip(text_tokens, text_tokens_as_str)
        ]
        return mapping, [offset] * len(texts), text_tokens_as_str, text_tokens

    def _convert_non_null_to_one(self, lst):
        lst_one = []
        for item in lst:
            if item == 0:
                lst_one.append(0)
            else:
                lst_one.append(1)
        return lst_one

    def _map_token_index_to_wordpiece_index(self, a, b):
        is_equal_tensor = a == b
        is_equal_tensor = torch.nonzero(is_equal_tensor, as_tuple=False)
        is_equal_tensor = is_equal_tensor.flatten()
        return is_equal_tensor.tolist()

    def _create_dependency_tensor(
        self,
        text_tokens,
        text_tokens_as_str,
        mapping_token2wordpiece,
        mapping_token2wordpiece_offset,
    ):
        # create the dependency matrix
        dependency_tensor_of_tokens = self._calculate_dep_matrix(
            text_tokens, text_tokens_as_str
        )
        # create empty wordpiece based tensors
        dependency_tensor_of_wordpieces = torch.zeros(
            self.max_seq_len, self.max_seq_len, dtype=torch.float
        )

        for word_index_withoffset, word in enumerate(
            text_tokens_as_str, start=mapping_token2wordpiece_offset
        ):
            word_index = word_index_withoffset - mapping_token2wordpiece_offset
            dependency_column = dependency_tensor_of_tokens[word_index].tolist()
            dependency_column_ones = self._convert_non_null_to_one(dependency_column)
            _check_num_heads = sum(dependency_column_ones)
            assert _check_num_heads in [
                0,
                1,
            ], (
                f"expected only zero or one heads, found {_check_num_heads} for word "
                f"{word} in {text_tokens_as_str}"
            )
            if _check_num_heads == 0:
                # this can happen if the head is in the part of the sentence that was
                # truncated due to max seq len
                logger.warning(
                    "head of word %s is not part of the wordpiece sequence", word
                )
                continue

            head_index = dependency_column_ones.index(1)
            head_index_withoffset = head_index + mapping_token2wordpiece_offset
            dependency_type = dependency_column[head_index]
            assert dependency_type != 0

            positions_of_current_word_in_input_sequence = (
                self._map_token_index_to_wordpiece_index(
                    mapping_token2wordpiece, word_index_withoffset
                )
            )
            positions_of_current_head_in_input_sequence = (
                self._map_token_index_to_wordpiece_index(
                    mapping_token2wordpiece, head_index_withoffset
                )
            )

            for position_word in positions_of_current_word_in_input_sequence:
                for position_head in positions_of_current_head_in_input_sequence:
                    dependency_tensor_of_wordpieces[
                        position_word, position_head
                    ] = dependency_type

        return dependency_tensor_of_wordpieces

    def _create_dependency_tree_hop_distances_of_tokens_to_target(
        self,
        text_tokens,
        text_left_len,
        target_phrase,
        text_tokens_as_str,
        mapping_token2wordpiece,
        mapping_token2wordpiece_offset,
    ):
        """
        finds for each token in the sentence, or rather each token's wordpiece(s) the distance in the
        syntax tree to the target phrase. returns a list of size k where k is the maxseqlen; the list
        contains the distances for each wordpiece.
        :param text_tokens:
        :param text_left_len:
        :param target_phrase:
        :param text_tokens_as_str:
        :param mapping_token2wordpiece:
        :param mapping_token2wordpiece_offset:
        :return:
        """
        # Find distance in dependency parsing tree
        num_chars_counted = sum([len(token) for token in text_tokens_as_str])
        dist = self._calculate_dep_distance(text_tokens, text_left_len, target_phrase)

        # create empty wordpiece based tensors
        syntax_hop_distance_to_target = torch.zeros(self.max_seq_len, dtype=torch.long)

        # convert:
        # syntax_hop_distance_to_target and dependency_tensor_of_wordpieces is wordpiece
        # based. dist and dependency_tensor_of_tokens is token based. map from each to
        # the wordpiece variant
        for word_index, word in enumerate(
            text_tokens_as_str, start=mapping_token2wordpiece_offset
        ):
            word_index_without_offset = word_index - mapping_token2wordpiece_offset

            positions_of_current_word_in_input_sequence = (
                self._map_token_index_to_wordpiece_index(
                    mapping_token2wordpiece, word_index
                )
            )

            depdistance_of_current_word_in_input_sequence = dist[
                word_index_without_offset
            ]

            for position in positions_of_current_word_in_input_sequence:
                syntax_hop_distance_to_target[
                    position
                ] = depdistance_of_current_word_in_input_sequence

        return syntax_hop_distance_to_target

    def create_model_input_seqs(
        self,
        text_left: str,
        target_phrase: str,
        text_right: str,
        coreferential_targets_for_target_mask: Optional[Iterable[dict]],
    ) -> Mapping[str, Mapping[str, Union[torch.Tensor, bool]]]:
        """
        Creates input sequences for a given target. Prior components have processed the jsonl
        so that coreferential_targets_for_target_mask will contain a list of coreferential
        mentions of the target if and only if coref_mode == "in_targetmask". In that case,
        will produce an individual target mask for each coref mention and merge them to
        the target mask of the preferred mention.
        """
        batch_result = self.batch_create_model_input_seqs(
            ((text_left, target_phrase, text_right),),
            coreferential_targets_for_target_mask=(
                coreferential_targets_for_target_mask,
            ),
        )
        return {
            tok_name: {
                seq_name: seq_result[0] for seq_name, seq_result in tok_result.items()
            }
            for tok_name, tok_result in batch_result.items()
        }

    @overload
    def batch_create_model_input_seqs(
        self,
        targets: ...,
        coreferential_targets_for_target_mask: ...,
    ) -> Mapping[str, Mapping[str, Union[torch.Tensor, Tuple[bool], bool]]]:
        ...

    @overload
    def batch_create_model_input_seqs(
        self,
        targets: ...,
        coreferential_targets_for_target_mask: ...,
    ) -> Mapping[str, Mapping[str, Union[torch.Tensor, bool]]]:
        ...

    def batch_create_model_input_seqs(
        self,
        targets: Sequence[Tuple[str, str, str]],
        coreferential_targets_for_target_mask: Optional[
            Sequence[Optional[Iterable[dict]]]
        ],
    ) -> Mapping[str, Mapping[str, Union[torch.Tensor, Tuple[bool], bool]]]:
        """
        Creates input sequences for given targets. Prior components have processed the jsonl
        so that coreferential_targets_for_target_mask will contain a list of coreferential
        mentions of the targets if and only if coref_mode == "in_targetmask". In that case,
        will produce an individual target mask for each coref mention and merge them to
        the target mask of the preferred mention.

        NEW
        Multiple target inputs. Output will be a tuple of batches with the
        created sequences being of shape [batch_size,seq_length]. To output a single
        target without batch output and with tensors of shape [seq_length],
        set single_target_output = True.
        """
        num_targets = len(targets)

        if not coreferential_targets_for_target_mask:
            coreferential_targets_for_target_mask = [None] * num_targets
        elif num_targets != len(coreferential_targets_for_target_mask):
            raise TypeError(
                "Number of coreferential_targets_for_target_mask must match number of targets."
            )

        lefts = []
        target_phrases = []
        adjusted_target_phrases = []
        full_texts = []

        for target in targets:
            assert all(isinstance(arg, str) for arg in target), "Wrong input types."

            left, target_phrase, _ = target

            lefts.append(left)

            target_phrases.append(target_phrase)

            adjusted_target_phrases.append(
                f"What do you think of {target_phrase}?"
                if self.is_use_natural_target_phrase_for_spc
                else target_phrase
            )

            full_texts.append(
                self.create_entire_text(*target, is_return_modified_text_left=False)
            )

        out: Mapping[str, Mapping[str, Union[torch.Tensor, bool]]] = {}

        for tok_name, tok_obj in self.tokenizers_name_and_obj.items():
            logger.debug(f"{tok_name}")

            # text
            text_ids_with_special_tokens_per_target = tok_obj(
                text=full_texts,
                max_length=self.max_seq_len,
                padding="max_length",
                add_special_tokens=True,
                truncation="longest_first",
            )["input_ids"]
            text_then_target_ids_with_special_tokens_dict_per_target = (
                tok_obj.batch_encode_plus(
                    zip(full_texts, adjusted_target_phrases),
                    max_length=self.max_seq_len,
                    padding="max_length",
                    add_special_tokens=True,
                    truncation="longest_first",
                )
            )
            text_then_target_ids_with_special_tokens_dict_per_target = (
                text_then_target_ids_with_special_tokens_dict_per_target.data
            )
            text_then_target_ids_with_special_tokens_per_target = (
                text_then_target_ids_with_special_tokens_dict_per_target["input_ids"]
            )
            if type(tok_obj) == RobertaTokenizer:
                # roberta doesnt have segments, so we produce fake segment ids
                # here, and later simply dont pass them
                text_then_target_ids_with_special_tokens_segment_ids_per_target = [
                    [0 for _ in target_ids]
                    for target_ids in text_then_target_ids_with_special_tokens_per_target
                ]
            else:
                text_then_target_ids_with_special_tokens_segment_ids_per_target = (
                    text_then_target_ids_with_special_tokens_dict_per_target[
                        "token_type_ids"
                    ]
                )
            # target
            target_ids_with_special_tokens_per_target = tok_obj(
                target_phrases,
                max_length=self.max_seq_len,
                padding="max_length",
                add_special_tokens=True,
                truncation="longest_first",
            )["input_ids"]

            # create target masks
            target_mask_seq_for_text_with_special_tokens_per_target = (
                self._batch_create_target_mask(
                    tok_obj, targets, for_text_with_special_tokens=True
                )
            )
            # create target mask for coreferential targets of given target
            coref_target_masks_per_target = (
                self._batch_create_coreferential_target_masks(
                    tok_obj, coreferential_targets_for_target_mask
                )
            )
            # merge them into the preferred target mask
            merged_target_mask_per_target = tuple(
                self._merge_coref_target_masks_into_preferred_target_mask(
                    target_mask,
                    coref_target_masks,
                )
                for target_mask, coref_target_masks in zip(
                    target_mask_seq_for_text_with_special_tokens_per_target,
                    coref_target_masks_per_target,
                )
            )
            target_mask_seq_for_text_with_special_tokens_per_target = (
                merged_target_mask_per_target
            )

            # create also text ids without max length to see if the one that will be
            # used was truncated
            text_ids_with_special_tokens_no_max_length_per_target = tok_obj(
                full_texts, add_special_tokens=True
            )["input_ids"]
            text_num_truncated_tokens_per_target = tuple(
                len(noml) - len(ml)
                for noml, ml in zip(
                    text_ids_with_special_tokens_no_max_length_per_target,
                    text_ids_with_special_tokens_per_target,
                )
            )

            # iterate tokens of full text and look up in dicts
            # create mapping from token-based indexes to wordpiece-based indexes
            # the function will also remove any tokens for which the word piece index
            # would be after max seq len
            # under certain circumstances, e.g., when using albert as well as the target
            # phrase is close to what would be cut off (but is not at this point), it can happen
            # that at a later point in time, i.e., _create_dependency_tree_hop_distances_of_tokens_to_target
            # or more specifically _calculate_dep_distance, the target cannot be found in the list of
            # tokens anymore, because this list of now cut off. thus, i changed the assert in
            # _calculate_dep_distance to throwing an exception that is catched later on
            (
                mapping_token2wordpiece_per_target,
                mapping_token2wordpiece_offset_per_target,
                text_tokens_as_str_per_target,
                text_tokens_per_target,
            ) = self._batch_create_mapping_from_tokenbased_to_wordpiece_based(
                full_texts, tok_obj
            )

            text_stacked_knowledge_source_info_per_target = []
            text_dependency_tree_hop_distances_per_target = []
            text_dependency_matrix_per_target = []

            for (
                target_phrase,
                left,
                mapping_token2wordpiece,
                mapping_token2wordpiece_offset,
                text_tokens_as_str,
                text_tokens,
            ) in zip(
                target_phrases,
                lefts,
                mapping_token2wordpiece_per_target,
                mapping_token2wordpiece_offset_per_target,
                text_tokens_as_str_per_target,
                text_tokens_per_target,
            ):
                # create additional knowledge source tensors
                # stack only those knowledge sources that were requested by arguments
                selected_tensors_knowledge_sources_text = []
                for source in self.knowledge_sources:
                    text_tensor = self._create_knowledge_source_tensor(
                        text_tokens_as_str,
                        mapping_token2wordpiece_offset,
                        mapping_token2wordpiece,
                        source,
                    )
                    selected_tensors_knowledge_sources_text.append(text_tensor)
                text_stacked_knowledge_source_info_per_target.append(
                    torch.cat(tuple(selected_tensors_knowledge_sources_text), dim=1)
                    if self.knowledge_sources
                    else None
                )

                # syntax hop distance
                text_dependency_tree_hop_distances_per_target.append(
                    self._create_dependency_tree_hop_distances_of_tokens_to_target(
                        text_tokens,
                        len(left),
                        target_phrase,
                        text_tokens_as_str,
                        mapping_token2wordpiece,
                        mapping_token2wordpiece_offset,
                    )
                )

                text_dependency_matrix_per_target.append(
                    self._create_dependency_tensor(
                        text_tokens,
                        text_tokens_as_str,
                        mapping_token2wordpiece,
                        mapping_token2wordpiece_offset,
                    )
                )

            # create item with indexes and masks
            result = {
                FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS: torch.LongTensor(
                    text_ids_with_special_tokens_per_target
                ),
                FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK: torch.FloatTensor(
                    target_mask_seq_for_text_with_special_tokens_per_target
                ),
                FIELD_IS_OVERFLOW: tuple(
                    text_num > 0 for text_num in text_num_truncated_tokens_per_target
                ),
                FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS: torch.LongTensor(
                    text_then_target_ids_with_special_tokens_per_target
                ),
                FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_SEGMENT_IDS: torch.LongTensor(
                    text_then_target_ids_with_special_tokens_segment_ids_per_target
                ),
                FIELD_TARGET_IDS_WITH_SPECIAL_TOKENS: torch.LongTensor(
                    target_ids_with_special_tokens_per_target
                ),
                FIELD_SYNTAX_HOP_DISTANCE_TO_TARGET: torch.stack(
                    text_dependency_tree_hop_distances_per_target
                ),
                FIELD_SYNTAX_DEPENDENCY_MATRIX: torch.stack(
                    text_dependency_matrix_per_target
                ),
            }

            # add knowledge source if requested
            if self.knowledge_sources:
                result[
                    FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_SELECTED_KNOWLEDGE_SOURCES
                ] = torch.stack(text_stacked_knowledge_source_info_per_target)

            # add tokenizer result to output
            out[tok_name] = result

        return out

    def _create_coreferential_target_masks(
        self, tok_obj, coreferential_targets_for_target_mask: Optional[Iterable[dict]]
    ):
        return self._batch_create_coreferential_target_masks(
            tok_obj, (coreferential_targets_for_target_mask,)
        )[0]

    def _batch_create_coreferential_target_masks(
        self,
        tok_obj,
        coreferential_targets_for_target_mask_per_target: Sequence[
            Optional[Iterable[dict]]
        ],
    ):
        target_masks_per_target = []
        for coreferential_targets in coreferential_targets_for_target_mask_per_target:
            target_mask = []
            if coreferential_targets:
                coref_targets = [
                    (
                        self.prepare_left_segment(coref_target["text_left"]),
                        self.prepare_target_mention(coref_target["mention"]),
                        "",
                    )
                    for coref_target in coreferential_targets
                ]
                target_mask.append(
                    self._batch_create_target_mask(
                        tok_obj, coref_targets, for_text_with_special_tokens=True
                    )
                )
            target_masks_per_target.append(target_mask)
        return target_masks_per_target

    def _merge_coref_target_masks_into_preferred_target_mask(
        self, target_mask_seq_for_text_with_special_tokens, coref_target_masks
    ):
        if coref_target_masks is None or len(coref_target_masks) == 0:
            return target_mask_seq_for_text_with_special_tokens

        # target_mask_seq_for_text_with_special_tokens is a list of scalars, coref tms a list of lists
        # turn both into a list of lists and append them
        all_target_masks = [
            target_mask_seq_for_text_with_special_tokens
        ] + coref_target_masks
        assert len(all_target_masks) == 1 + len(coref_target_masks)
        num_scalars = len(target_mask_seq_for_text_with_special_tokens)
        # ensure all same length
        for lst in all_target_masks:
            assert len(lst) == num_scalars
        # create merged lst
        merged_target_mask = [0] * num_scalars
        for index in range(num_scalars):
            merged_value = 0
            for lst in all_target_masks:
                lst_cur_value = lst[index]
                assert lst_cur_value in [0, 1]
                merged_value += lst_cur_value
            merged_value = min(1, merged_value)
            merged_target_mask[index] = merged_value

        return merged_target_mask


class FXDataset(Dataset):
    __PARAMS2INDEX = {}
    NUM_MAX_TARGETS_PER_ITEM = 2
    SINGLE_TARGETS = False

    def __init__(
        self,
        filepath,
        data_format: str,
        tokenizer: FXEasyTokenizer,
        named_polarity_to_class_number,
        class_number_to_named_polarity,
        sorted_expected_label_names,
        single_targets: bool,
        coref_mode: str,
        devmode=False,
        ignore_parsing_errors=False,
    ):
        self.class_label2class_number = named_polarity_to_class_number
        self.class_number2class_label = class_number_to_named_polarity
        self.sorted_expected_label_names = sorted_expected_label_names
        self.tokenizer = tokenizer
        self.data = []
        self.data_format = data_format
        self.count_sequence_overflow = Counter()
        FXDataset.SINGLE_TARGETS = single_targets

        if FXDataset.SINGLE_TARGETS:
            # single mode is enabled, so there will be only 1 target per item
            FXDataset.NUM_MAX_TARGETS_PER_ITEM = 1

        logger.info("reading dataset file {}".format(filepath))

        tasks = []
        with jsonlines.open(filepath, "r") as reader:
            for task in reader:
                task["sentence_normalized"] = self.__prepare_text_field(
                    task["sentence_normalized"]
                )
                for target in task["targets"]:
                    target["mention"] = self.__prepare_text_field(target["mention"])
                    further_mentions = []
                    for further_mention in target.get("further_mentions", []):
                        further_mention["mention"] = self.__prepare_text_field(
                            further_mention["mention"]
                        )
                        further_mentions.append(further_mention)
                    target["further_mentions"] = further_mentions
                tasks.append(task)

        if devmode:
            k = 5
            logger.warning("DEV MODE IS ENABLED")
            logger.info("devmode=True: truncating dataset to {} lines".format(k))
            tasks = tasks[:k]

        self.label_counter = Counter()
        with tqdm(total=len(tasks)) as pbar:
            for task in tasks:
                # create model items from this jsonl row (in case of single_targets
                # enabled, a row with k targets will return k model items)
                model_items = self.task_to_dataset_item(
                    task, coref_mode, ignore_parsing_errors
                )
                for converted_item, count_labels in model_items:
                    if converted_item is None:
                        # this will happen if an example has 0 targets -> ignore
                        self.label_counter["ignored_example"] += 1
                    else:
                        self.label_counter += count_labels
                        self.data.append(converted_item)
                pbar.update(1)
        logger.info("label distribution: {}".format(self.label_counter))

    def __prepare_text_field(self, text: str):
        """
        Replaces "invalid" characters with something else of the same length (same
        length is important so that the offsets in the dataset are still valid)
        :param text:
        :return:
        """
        return text.replace("\xa0", " ").replace("��", "  ")

    def __create_target_text_components(
        self,
        target_start_char,
        target_end_char,
        text,
        check_mention,
        ignore_parsing_errors,
    ):
        target_check_mention = text[target_start_char:target_end_char]
        text_left = text[:target_start_char]
        text_right = text[target_end_char:]
        # perform target mention check
        if ignore_parsing_errors:
            if check_mention != target_check_mention:
                logger.warning(
                    f"indexes do not match: '{check_mention}' vs '{target_check_mention}' "
                    f"({text_left}; {text_right}; {target_start_char}; {target_end_char})"
                )
        else:
            assert check_mention == target_check_mention, (
                f"indexes do not match: '{check_mention}' vs '{target_check_mention}' "
                f"({text_left}; {text_right}; {target_start_char}; {target_end_char})"
            )

        return text_left, text_right

    def __parse_dataset_row(self, task, ignore_parsing_errors):
        if self.data_format == "newsmtsc":
            _text = task["sentence_normalized"]
            primary_mtsc_gid = task["primary_gid"]
            if not FXDataset.SINGLE_TARGETS:
                # if single_targets mode is not enabled, ensure each item has at most
                # FXDataset.NUM_MAX_TARGETS_PER_ITEM targets
                # assert 1 <= len(task["targets"]) <= FXDataset.NUM_MAX_TARGETS_PER_ITEM
                pass
            else:
                # if single_targets mode is enabled, FXDataset.NUM_MAX_TARGETS_PER_ITEM
                # was set to 1, so we cannot perform the check at this point. and also
                # do not need to ensure because later we will expand the items so that
                # resulting items each will only have 1 target
                pass

            # perform validity check of target mention
            # also, add text_left and text_right to each target
            for target in task["targets"]:
                target_ae_gid = target["Input.gid"]
                target_start_char = target["from"]
                target_end_char = target["to"]
                target_mention = target["mention"]
                target_polarity = target["polarity"]

                text_left, text_right = self.__create_target_text_components(
                    target_start_char,
                    target_end_char,
                    _text,
                    target_mention,
                    ignore_parsing_errors,
                )

                # add target-specific text_left and text_right
                target["text_left"] = text_left
                target["text_right"] = text_right

            targets = task["targets"]
            return primary_mtsc_gid, _text, targets
        else:
            raise ValueError(f"data format unknown: {self.data_format}")

    def _create_target_inputs(
        self,
        text_left: str,
        target_mention: str,
        text_right: str,
        target_gid: str,
        polarity: float,
        count_labels: Counter,
        coreferential_targets_for_target_mask: Iterable[dict],
        is_fillup: bool,
    ):
        # always convert polarity to float
        polarity = float(polarity)

        # earlier, we assumed basic tokanization, e.g., that "Mr. Smith's statement
        # is stupid." was already tokenized into "Mr. Smith 's statement is stupid ."
        # however, since both encode and encode_plus internally perform tokenization
        # when invoked with a str object, we do not need to assume that the input is
        # already tokenized.
        # here is how we handle the input text and target phrase
        # 1) we ensure that there is a whitespace between left and target. if there
        #    is no whitespace, we add a whitespace to the end of left
        # 2) we do the same as in (1) for target and right, and if necessary add a
        #    whitespace to the beginning of right
        # by doing these two operations, we ensure that any target-dependent mapping
        # e.g., the target mask, can be mapped 1:1 to the the input text. for
        # example, "<target>Mr. Smith</target>'s statement is stupid." will be
        # converted into "Mr. Smith 's statement ..." this way, the LM's encode or
        # encode_plus function will (after internal tokenization) yield a list of
        # input indexes that we can unambiguously map to the target mask that we
        # produce ourselves, e.g., in FXEasyTokenizer:_create_target_mask. said
        # differently, by adding the whitespace, we ensure that "Mr. Smith" will be
        # mapped to its own indexes, whereas if we passed "Mr. Smith's", this might
        # yield different indexes, e.g., if "Smith's" is part of the LM's
        # tokenizer's vocabulary. for more information, see NA-633 and NA-643.
        text_left = FXEasyTokenizer.prepare_left_segment(text_left)
        target_mention = FXEasyTokenizer.prepare_target_mention(target_mention)
        text_right = FXEasyTokenizer.prepare_right_segment(text_right)
        text = FXEasyTokenizer.create_entire_text(
            text_left,
            target_mention,
            text_right,
            is_return_modified_text_left=False,
        )

        # text to indexes
        try:
            inputs_of_target = self.tokenizer.create_model_input_seqs(
                text_left,
                target_mention,
                text_right,
                coreferential_targets_for_target_mask,
            )
        except TooLongTextException:
            logger.error(
                "text is too long for target mask creation for target_gid=%s, ignoring target",
                target_gid,
            )
            return None
        except TargetNotFoundException as tnfe:
            logger.error(
                "target was not found after preprocessing when calculating "
                "distance hop for target_gid=%s, ignoring target (error msg: %s)",
                target_gid,
                str(tnfe),
            )
            return None

        # count overflow
        for tokenizer_name in self.tokenizer.tokenizers_name_and_obj.keys():
            if inputs_of_target[tokenizer_name]["is_overflow"]:
                self.count_sequence_overflow[tokenizer_name] += 1
                self.count_sequence_overflow["all"] += 1

        # add polarity
        inputs_of_target["polarity"] = SentimentClasses.polarity2normalized_polarity(
            polarity
        )
        inputs_of_target["label"] = SentimentClasses.polarity2label(polarity)
        # update stats
        count_labels[inputs_of_target["label"]] += 1

        # add reference information
        inputs_of_target["target_gid"] = target_gid

        # add original text and target (we can use that to create a mistake table)
        inputs_of_target["orig_text"] = text
        inputs_of_target["orig_target"] = target_mention
        inputs_of_target["orig_polarity"] = polarity

        # add whether fill up target (fake) or real
        inputs_of_target["is_fillup"] = is_fillup

        return inputs_of_target

    def _create_fillup_target(self):
        return self._create_target_inputs(
            text_left=" ",
            target_mention="fillup",
            text_right=" ",
            target_gid="fillup",
            polarity=SentimentClasses.FILLUP_POLARITY_VALUE,
            count_labels=Counter(),
            coreferential_targets_for_target_mask=[],
            is_fillup=True,
        )

    def _get_as_list(self, lst_inputs_of_targets: List[dict], key: str):
        """
        selects the value under the key in each
        list element of lst_inputs_of_targets. returns these values as a new list
        :param lst_inputs_of_targets:
        :param key:
        :return:
        """
        new_list = []
        for inputs_of_targets in lst_inputs_of_targets:
            new_list.append(inputs_of_targets[key])
        return new_list

    def _get_values_as_lists_in_dict(
        self, lst_inputs_of_targets: List[dict], keys: Iterable[str]
    ):
        d = {}
        for key in keys:
            d[key] = self._get_as_list(lst_inputs_of_targets, key)
        return d

    def _convert_multi_targets_in_single_item_to_single_target_in_multi_items(
        self, example_id, text, targets
    ):
        """
        Converts a single item with k targets into k items with 1 target each
        :param example_id:
        :param text:
        :param targets:
        :return: list of tuples (example_id, text, targets)
        """
        single_items = []
        for target in targets:
            single_example_id = target["Input.gid"]
            single_text = text
            single_targets = [target]
            single_item = (single_example_id, single_text, single_targets)
            single_items.append(single_item)
        return single_items

    def _convert_multi_targets_in_single_item_to_k_targets_in_multi_items(
        self, example_id, text, targets
    ):
        """
        Converts a single item with m targets into ceil(m/k) items with k items. If a
        resulting item would have less than k targets, its existing targets are repeated
        so that each item will have exactly k targets (in this case with duplicate
        targets).
        :param example_id:
        :param text:
        :param targets:
        :return:
        """
        shuffled_targets = targets.copy()
        random.shuffle(shuffled_targets)
        num_targets_per_item = FXDataset.NUM_MAX_TARGETS_PER_ITEM
        logger.debug(shuffled_targets)

        items = []
        count_processed_targets = 0
        for i in range(0, len(shuffled_targets), num_targets_per_item):
            cur_targets = shuffled_targets[i : i + num_targets_per_item]
            if len(cur_targets) < num_targets_per_item:
                logger.debug("too few targets")
                break
            assert len(cur_targets) == num_targets_per_item
            virtual_example_id = example_id + "_" + str(i)
            items.append((virtual_example_id, text, cur_targets))
            count_processed_targets += len(cur_targets)

        # process remaining targets
        logger.debug("processing remaining targets")
        if count_processed_targets < len(shuffled_targets):
            remaining_targets = shuffled_targets[count_processed_targets:]
            assert len(remaining_targets) < num_targets_per_item

            # duplicate targets until we have enough targets
            # create intermediate list to not skew random selection because of adding
            # the duplicates immediately
            tmp_targets = []
            while len(remaining_targets) + len(tmp_targets) < num_targets_per_item:
                tmp_targets.append(random.choice(remaining_targets))
                logger.debug(
                    f"{len(remaining_targets)}, {len(tmp_targets)}, {num_targets_per_item}"
                )
            remaining_targets.extend(tmp_targets)
            assert len(remaining_targets) == num_targets_per_item
            virtual_example_id = example_id + "_last"
            items.append((virtual_example_id, text, remaining_targets))

        # validation
        count_targets = 0
        seen_targets = []
        for item in items:
            targets = item[2]
            count_targets += len(targets)
            for target in targets:
                seen_targets.append(target)
        assert count_targets % num_targets_per_item == 0
        # seen targets must either be of same length as shuffled target, or - in case of
        # added duplicate targets - larger
        assert len(seen_targets) >= len(
            shuffled_targets
        ), f"{len(seen_targets)} vs. {len(shuffled_targets)}"

        return items

    def _create_item_for_model(self, example_id, text, targets):
        lst_inputs_of_targets = []
        count_labels = Counter()

        for target in targets:
            text_left = target["text_left"]
            text_right = target["text_right"]
            target_mention = target["mention"]
            polarity = target["polarity"]
            target_gid = target["Input.gid"]
            coreferential_targets_for_target_mask = target.get(
                "coreferential_targets_for_target_mask", []
            )

            inputs_of_target = self._create_target_inputs(
                text_left=text_left,
                target_mention=target_mention,
                text_right=text_right,
                target_gid=target_gid,
                polarity=polarity,
                count_labels=count_labels,
                coreferential_targets_for_target_mask=coreferential_targets_for_target_mask,
                is_fillup=False,
            )
            if inputs_of_target is None:
                # this will happen if a target mask could not be created because the
                # text_left is too long. -> ignore this target
                pass
            else:
                lst_inputs_of_targets.append(inputs_of_target)
        num_actual_targets = len(lst_inputs_of_targets)

        if num_actual_targets == 0:
            logger.error(
                "ignoring example (gid=%s) because it has 0 targets", example_id
            )
            return None, None

        # due to torch's requirements in dataloader, we need fill up remaining targets
        # with fake targets
        while len(lst_inputs_of_targets) < FXDataset.NUM_MAX_TARGETS_PER_ITEM:
            lst_inputs_of_targets.append(self._create_fillup_target())

        # lastly, merge inputs to one tensor over all targets
        # this also removes some fields, e.g., list of strs. should we need the removed
        # information later, we could simply add a new field below to "converted_item"
        stacked_targets, skipped_keys = self._stack_target_inputs(lst_inputs_of_targets)

        # create item dict
        # originally, we also had original targets here, but they get scrambled up and
        # there is no StrTensor to easily preserve their order
        converted_item = {
            "text": text,
            "num_targets": num_actual_targets,
            "primary_gid": example_id,
        }

        # merge stacked_targets dict into item
        # first, though, ensure no overlap by key
        assert len(set(converted_item.keys()).intersection(stacked_targets.keys())) == 0
        # now merge
        converted_item = {**converted_item, **stacked_targets}

        return converted_item, count_labels

    def create_virtual_target_from_target_mention(self, target, mention, text):
        mention_from = mention["from"]
        mention_to = mention["to"]
        mention_text = mention["mention"]

        mention_gid = target["Input.gid"] + "__" + mention_text
        mention_polarity = target["polarity"]

        text_left, text_right = self.__create_target_text_components(
            mention_from, mention_to, text, mention_text, False
        )

        return {
            "Input.gid": mention_gid,
            "from": mention_from,
            "to": mention_to,
            "mention": mention_text,
            "polarity": mention_polarity,
            "text_left": text_left,
            "text_right": text_right,
        }

    def _expand_coref_mentions_of_targets_to_multiple_targets(
        self, targets: Iterable, text: str
    ):
        expanded_targets = []
        for target in targets:
            expanded_targets.append(target)

            further_mentions = target.get("further_mentions", [])
            for further_mention in further_mentions:
                virtual_target = self.create_virtual_target_from_target_mention(
                    target, further_mention, text
                )
                expanded_targets.append(virtual_target)

        return expanded_targets

    def task_to_dataset_item(self, task, coref_mode: str, ignore_parsing_errors):
        # parse current dataset row (=task)
        example_id, text, targets = self.__parse_dataset_row(
            task, ignore_parsing_errors
        )

        # if the coref mode (only during training though) is additional_examples, repeat them here
        if coref_mode == "additional_examples":
            expanded_targets = (
                self._expand_coref_mentions_of_targets_to_multiple_targets(
                    targets, text
                )
            )
        elif coref_mode == "in_targetmask":
            expanded_targets = targets
            for target in expanded_targets:
                further_mentions = target.get("further_mentions", [])
                targets_for_target_mask = []
                for further_mention in further_mentions:
                    virtual_target = self.create_virtual_target_from_target_mention(
                        target, further_mention, text
                    )
                    targets_for_target_mask.append(virtual_target)
                target[
                    "coreferential_targets_for_target_mask"
                ] = targets_for_target_mask
        else:
            # if coref mode is ignore, nothing to do here
            expanded_targets = targets

        # if single_targets mode is enabled, we want to have only one target for each
        # sentence, thus expand sentences
        if FXDataset.SINGLE_TARGETS:
            items = self._convert_multi_targets_in_single_item_to_single_target_in_multi_items(
                example_id, text, expanded_targets
            )
        else:
            items = (
                self._convert_multi_targets_in_single_item_to_k_targets_in_multi_items(
                    example_id, text, expanded_targets
                )
            )

        # iterate (virtual) items of this actual row in the jsonl and create model items
        model_items = []
        for example_id, text, targets in items:
            res = self._create_item_for_model(example_id, text, targets)
            model_items.append(res)

        # return
        return model_items

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def set_params2index(params2index: dict):
        if len(FXDataset.__PARAMS2INDEX) > 0:
            # verify the new one is same
            assert FXDataset.__PARAMS2INDEX == params2index
        else:
            FXDataset.__PARAMS2INDEX = params2index

    @staticmethod
    def get_input_by_params(inputs: List, weight_name: str, field_name: str):
        """
        If FXDataset.SINGLE_TARGETS is True, include the target dimension (currently 2nd
        dimension in tensor (dims are: batch, target, indexes or other values). If
        False, remove the target dimension.
        :param inputs:
        :param weight_name:
        :param field_name:
        :param include_target_dimension:
        :return:
        """
        param_id = (weight_name, field_name)
        seq = inputs[FXDataset.__PARAMS2INDEX[param_id]]
        if not FXDataset.SINGLE_TARGETS:
            return seq
        else:
            assert seq.shape[1] == 1, "there are more than one targets"
            return seq.squeeze(dim=1)

    @staticmethod
    def get_all_inputs_for_model(inputs: List, model_class: FXBaseModel):
        selected_inputs = []
        for weight_name, field_name in model_class.get_input_field_ids():
            selected_inputs.append(
                FXDataset.get_input_by_params(inputs, weight_name, field_name)
            )
        return inputs

    def _stack_target_inputs(self, lst_inputs_of_targets: List[dict]) -> dict:
        """
        stacks each field to one Tensor
        :param lst_inputs_of_targets:
        :return:
        """
        # ensure correct number of targets
        assert len(lst_inputs_of_targets) == self.NUM_MAX_TARGETS_PER_ITEM
        # ensure each target has same keys
        all_keys = list(lst_inputs_of_targets[0].keys())
        for inputs_of_target in lst_inputs_of_targets:
            assert set(all_keys) == inputs_of_target.keys()
        # for each key, merge all targets
        merged_targets = {}

        # to collect keys that were skipped
        skipped_keys = []

        # keys that not "scalars", e.g., polarity, but dicts, e.g., "bert-base-uncased",
        # which itself holds multiple vectors, require special handling
        special_keys = list(self.tokenizer.tokenizers_name_and_obj.keys())
        # iterate keys
        for key in all_keys:
            if key in special_keys:
                merged_targets[key] = self._stack_special_field(
                    key, lst_inputs_of_targets
                )
            else:
                inputs_over_targets_of_key = []
                for inputs_of_target in lst_inputs_of_targets:
                    inputs_over_targets_of_key.append(inputs_of_target[key])

                # merge targets' inputs to one tensor
                stacked_inputs_over_targets_of_key = self._stack_or_create_tensor(
                    inputs_over_targets_of_key
                )
                # add stacked tensor to dict
                if stacked_inputs_over_targets_of_key is not None:
                    merged_targets[key] = stacked_inputs_over_targets_of_key
                else:
                    # if None, dont add
                    logger.debug("skipping key: %s", key)
                    skipped_keys.append(key)

        return merged_targets, skipped_keys

    def _stack_special_field(self, key: str, lst_inputs_of_targets: List[dict]):
        # ensure each target has same sub keys for current selected key
        all_sub_keys = list(lst_inputs_of_targets[0][key].keys())
        for inputs_of_target in lst_inputs_of_targets:
            assert set(all_sub_keys) == inputs_of_target[key].keys()

        # for each sub key, merge all targets
        merged_targets = {}

        for sub_key in all_sub_keys:
            inputs_over_targets_of_sub_key = []
            for inputs_of_target in lst_inputs_of_targets:
                # select
                inputs = inputs_of_target[key][sub_key]
                inputs_over_targets_of_sub_key.append(inputs)
            # merge targets' inputs to one tensor
            stacked_inputs_over_targets_of_sub_key = self._stack_or_create_tensor(
                inputs_over_targets_of_sub_key
            )
            # add stacked tensor to dict
            if stacked_inputs_over_targets_of_sub_key is not None:
                merged_targets[sub_key] = stacked_inputs_over_targets_of_sub_key
            else:
                # if None, dont add
                logger.debug("skipping key: %s", sub_key)

        return merged_targets

    def _stack_or_create_tensor(self, lst_values: List):
        """
        Given a list of values, checks if all are of same type and creates a single
        Tensor from the values (either by stacking or by creation)
        :param lst_values:
        :return:
        """
        # assert all same type
        first_type = type(lst_values[0])
        for val in lst_values:
            assert first_type == type(val), f"{first_type} vs {type(val)}"

        # stack list of tensors to one tensor
        if first_type == bool:
            # if bool, create a new tensor
            stacked_inputs_over_targets_of_sub_key = torch.BoolTensor(lst_values)
        elif first_type == int:
            # we need to have Long here, otherwise will run into an error during loss
            # calculation, where Long is expected
            stacked_inputs_over_targets_of_sub_key = torch.LongTensor(lst_values)
        elif first_type == float:
            stacked_inputs_over_targets_of_sub_key = torch.FloatTensor(lst_values)
        elif first_type == str:
            stacked_inputs_over_targets_of_sub_key = None
        elif torch.is_tensor(lst_values[0]):
            # if Tensor of subtype, stack
            stacked_inputs_over_targets_of_sub_key = torch.stack(lst_values)
        else:
            raise NotImplementedError(f"no merging for type: {first_type}")

        return stacked_inputs_over_targets_of_sub_key
