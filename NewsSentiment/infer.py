import argparse

import torch
import torch.nn.functional as F
from jsonlines import jsonlines
from tqdm import tqdm

from NewsSentiment.SentimentClasses import SentimentClasses
from NewsSentiment.dataset import FXEasyTokenizer
from NewsSentiment.download import Download
from NewsSentiment.fxlogger import get_logger
from NewsSentiment.models.singletarget.grutscsingle import GRUTSCSingle
from NewsSentiment.train import (
    parse_arguments as parse_arguments_from_train,
    prepare_and_start_instructor,
)


class TargetSentimentClassifier:
    def __init__(
        self,
        opts_from_infer=None,
        single_targets=True,
        logging_level="ERROR",
        state_dict=None,
    ):
        self.logger = get_logger()

        # reset NUM_CATEGORIES_OF_SELECTED_KNOWLEDGE_SOURCES
        FXEasyTokenizer.NUM_CATEGORIES_OF_SELECTED_KNOWLEDGE_SOURCES = 0

        # get the default parameters from train.py
        final_opts = parse_arguments_from_train(
            override_args=True, overwrite_logging_level=logging_level
        )
        # if no arguments have been passed, get the defaults of infer.py (and since
        # override_args=True, not the arguments passed by the console (in args) will be
        # used but just the defaults defined in parse_arguments
        if not opts_from_infer:
            opts_from_infer = parse_arguments(override_args=True)

        for key, val in vars(opts_from_infer).items():
            if val is not None:
                previous_val = getattr(final_opts, key)
                self.logger.info("overwriting: %s=%s to %s", key, previous_val, val)
                setattr(final_opts, key, val)

        # set custom variables
        final_opts.single_targets = single_targets
        final_opts.multi_targets = not single_targets
        if state_dict:
            final_opts.state_dict = state_dict

        # set training_mode to False so that we get the Instructor object
        final_opts.training_mode = False

        # prepare and initialize instructor
        instructor = prepare_and_start_instructor(final_opts)

        # get stuff that we need from the instructor
        self.model = instructor.own_model
        self.tokenizer = instructor.own_tokenizer
        self.opt = instructor.opt
        self.instructor = instructor
        self.polarities_inverse = SentimentClasses.get_polarity_associations_inverse()

        # set model to evaluation mode (disable gradient / learning)
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def infer_from_text(self, left, target, right):
        """
        Calculates the targeted mention of the sentence defined by left + target + right.
        :param left: The text from the beginning of the sentence to the mention of the
        target (will be empty if the sentence starts with the mention of the target).
        Make sure to include a space if there is one between the left phrase and the
        target mention, e.g., "I like Peter." -> left="I like ", target="Peter", right=".".
        :param target: The mention of the target.
        :param right: The text from the end of the target to the end of the sentence.
        Make sure to include a space if there is one between the target and the next word
        of the sentence.
        :return:
        """
        return self.infer(text_left=left, target_mention=target, text_right=right)

    def infer(
        self,
        text_left: str = None,
        target_mention: str = None,
        text_right: str = None,
        text: str = None,
        target_mention_from: int = None,
        target_mention_to: int = None,
    ):
        """
        Calculates sentiment as to target_mention in a text that is a concatenation of
        text_left,
        target_mention, and text_right. Note that text_left and text_right should end
        with a space (or comma, etc.)),
        or end with a space, respectively. Alternatively, the target can be selected via
        target_mention_from and target_mention_to in text.
        """
        is_index_based = (
            text is not None
            and target_mention_from is not None
            and target_mention_to is not None
        )
        is_component_based = (
            text_left is not None
            and target_mention is not None
            and text_right is not None
        )
        assert is_index_based != is_component_based

        if text:
            text_left = text[:target_mention_from]
            target_mention = text[target_mention_from:target_mention_to]
            text_right = text[target_mention_from:]

        # assert text_left.endswith(' ') # we cannot handle commas, if we have this
        # check
        assert not target_mention.startswith(" ") and not target_mention.endswith(
            " "
        ), f"target_mention={target_mention}; text={text}"
        # assert text_right.startswith(' ')

        text_left = FXEasyTokenizer.prepare_left_segment(text_left)
        target_mention = FXEasyTokenizer.prepare_target_mention(target_mention)
        text_right = FXEasyTokenizer.prepare_right_segment(text_right)

        indexed_example = self.tokenizer.create_model_input_seqs(
            text_left, target_mention, text_right, []
        )
        inputs = self.instructor.select_inputs(indexed_example, is_single_item=True)

        # invoke model
        outputs = self.model(inputs)
        class_probabilites = F.softmax(outputs, dim=-1).reshape((3,)).cpu().tolist()

        classification_result = []
        for class_id, class_prob in enumerate(class_probabilites):
            classification_result.append(
                {
                    "class_id": class_id,
                    "class_label": self.polarities_inverse[class_id],
                    "class_prob": class_prob,
                }
            )

        classification_result = sorted(
            classification_result, key=lambda x: x["class_prob"], reverse=True
        )

        return classification_result

    def get_info_for_label(self, classification_result, label):
        for r in classification_result:
            if r["class_label"] == label:
                return r
        raise ValueError(label)


def parse_arguments(override_args=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--own_model_name", default="grutsc", type=str)
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        help="has to be placed in folder pretrained_models",
    )
    parser.add_argument("--default_lm", default="roberta-base", type=str)
    parser.add_argument(
        "--state_dict", type=str, default=Download.model_path(GRUTSCSingle)
    )
    parser.add_argument(
        "--knowledgesources", default="nrc_emotions mpqa_subjectivity bingliu_opinion"
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="e.g., cuda:0; if None: any CUDA device will be used if available, else "
        "CPU",
    )
    parser.add_argument(
        "--logging", type=str, default="ERROR",
    )

    # if own_args == None -> parse_args will use sys.argv
    # if own_args == [] -> parse_args will use this empty list instead
    own_args = None
    if override_args:
        own_args = []

    # create arguments
    opt = parser.parse_args(args=own_args)

    return opt
