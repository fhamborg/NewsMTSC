import argparse

import torch
import torch.nn.functional as F
from math import ceil
from jsonlines import jsonlines
from tqdm import tqdm
from typing import overload, Any, List, Dict, Union, Sequence, Tuple, Optional

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

    @overload
    def infer(
        self,
        text_left: str = ...,
        target_mention: str = ...,
        text_right: str = ...,
        text: None = None,
        target_mention_from: None = None,
        target_mention_to: None = None,
        targets: None = None,
        batch_size: int = ...,
        disable_tqdm: bool = ...,
    ) -> Tuple[Dict[str, Any], ...]:
        ...

    @overload
    def infer(
        self,
        text_left: None = None,
        target_mention: None = None,
        text_right: None = None,
        text: str = ...,
        target_mention_from: int = ...,
        target_mention_to: int = ...,
        targets: None = None,
        batch_size: int = ...,
        disable_tqdm: bool = ...,
    ) -> Tuple[Dict[str, Any], ...]:
        ...

    @overload
    def infer(
        self,
        text_left: None = None,
        target_mention: None = None,
        text_right: None = None,
        text: None = None,
        target_mention_from: None = None,
        target_mention_to: None = None,
        targets: Sequence[Union[Tuple[str, str, str], Tuple[str, int, int]]] = ...,
        batch_size: int = ...,
        disable_tqdm: bool = ...,
    ) -> List[Tuple[Dict[str, Any], ...]]:
        ...

    def infer(
        self,
        text_left: Optional[str] = None,
        target_mention: Optional[str] = None,
        text_right: Optional[str] = None,
        text: Optional[str] = None,
        target_mention_from: Optional[int] = None,
        target_mention_to: Optional[int] = None,
        targets: Optional[
            Sequence[Union[Tuple[str, str, str], Tuple[str, int, int]]]
        ] = None,
        batch_size: int = 1,
        disable_tqdm: bool = False,
    ) -> Union[Tuple[Dict[str, Any], ...], List[Tuple[Dict[str, Any], ...]]]:
        """Computes sentiment for one or more targets. Additionally splits multiple
        targets into batches of batch_size and processes them in a vectorized fashion.

        Note that the text before and after the target should end with a space
        (or comma, etc.)), or begin with a space, respectively.

        Args:
            text_left (str | None, optional): Text before the target mention.
                Defaults to None.
            target_mention (str | None, optional): Target mention. Defaults to None.
            text_right (str | None, optional): Text after the target mention.
                Defaults to None.
            text (str | None, optional): Text containing the target mention.
                Defaults to None.
            target_mention_from (str | None, optional): Start index of the
                target mention. Defaults to None.
            target_mention_to (str | None, optional): End index of the target mention.
                Defaults to None.
            targets (Sequence[Tuple[str, str, str] | Tuple[str, int, int]] | None,
                optional): Tuples containing (text_left,target_mention,text_right)
                or (text,target_mention_from,target_mention_to) for multiple targets
                (mixed style is possible). Defaults to None.
            batch_size (int, optional): Preferred size of each batch if using multiple
                targets. Defaults to 1.
            disable_tqdm (bool, optional): Disables the tqdm progress bar that shows
                progress in batch processing. Defaults to False.

        Returns:
            Tuple[Dict[str, Any], ...] | List[Tuple[Dict[str, Any], ...]: Tuple (or
                list of tuples for multiple targets with order preserved) containing
                class probabilities as dictionaries with keys "class_id", "class_label"
                and "class_prob".
        """
        component_base = (text_left, target_mention, text_right)
        is_component_based = all(isinstance(i, str) for i in component_base)

        index_base = (text, target_mention_from, target_mention_to)
        is_index_based = all(
            isinstance(arg, typ) for arg, typ in zip(index_base, (str, int, int))
        )

        is_targets_based = targets is not None

        # verify input
        assert (
            sum((is_component_based, is_index_based, is_targets_based)) == 1
        ), """Wrong input types or too many inputs!
            Must be either one single or multiple component or index based targets."""

        targets = (
            (component_base if is_component_based else index_base,)
            if not is_targets_based
            else targets
        )

        num_targets = len(targets)
        num_batches = ceil(num_targets / batch_size)

        out = []

        for batch_start, batch_end in tqdm(
            zip(
                range(0, num_targets, batch_size),
                (*range(batch_size, num_targets, batch_size), None),
            ),
            total=num_batches,
            desc="Processing batches",
            unit="batch",
            disable=disable_tqdm,
        ):
            out.extend(self.batch_infer(targets[batch_start:batch_end]))

        return out[0] if not is_targets_based else out

    def batch_infer(
        self,
        targets: Sequence[Union[Tuple[str, str, str], Tuple[str, int, int]]],
    ) -> List[Tuple[Dict[str, Any], ...]]:
        """Computes sentiment for a batch of targets.
        Targets are tuples of text before target, target mention, and text after target.

        Args:
            targets (Sequence[Tuple[str,str,str] | Tuple[str,int,int]]): Batch of
                targets to compute sentiment for. Tuples contain
                (text before, target mention, text after) or
                (text,target mention start index, target mention end index).
                Texts before and after the target mention should end with a space
                (or comma, etc.)), or begin with a space, respectively.

        Returns:
            List[Tuple[Dict[str,Any], ...]]: List of target classification tuples,
                containing class probabilities as dictionaries with keys
                "class_id", "class_label" and "class_prob".
                The order of tuples matches the order of input targets.
        """
        targets_prepared = []

        for target in targets:
            assert (
                len(target) == 3
            ), f"{target} is missing {3-len(target)} component(s)."

            if all(isinstance(component, str) for component in target):
                text_left, target_mention, text_right = target
            elif all(
                isinstance(component, typ)
                for component, typ in zip(target, (str, int, int))
            ):
                text = target[0]
                target_mention_from = target[1]
                target_mention_to = target[2]
                text_left = text[:target_mention_from]
                target_mention = text[target_mention_from:target_mention_to]
                text_right = text[target_mention_to:]
            else:
                raise TypeError(f"Wrong input types in {target}.")

            # assert text_left.endswith(' ') # we cannot handle commas, if we have this
            # check
            assert not any(
                (target_mention.startswith(" "), target_mention.endswith(" "))
            ), f"target_mention={target_mention}; text={text}"
            # assert text_right.startswith(' ')

            targets_prepared.append(
                (
                    FXEasyTokenizer.prepare_left_segment(text_left),
                    FXEasyTokenizer.prepare_target_mention(target_mention),
                    FXEasyTokenizer.prepare_right_segment(text_right),
                )
            )

        indexed_examples = self.tokenizer.batch_create_model_input_seqs(
            targets=targets_prepared,
            coreferential_targets_for_target_mask=None,
        )

        inputs = self.instructor.select_inputs(indexed_examples, is_single_item=False)

        outputs = self.model(inputs)

        class_probabilities_per_target = F.softmax(outputs, dim=-1).cpu().tolist()

        return [
            tuple(
                sorted(
                    (
                        {
                            "class_id": class_id,
                            "class_label": self.polarities_inverse[class_id],
                            "class_prob": class_prob,
                        }
                        for class_id, class_prob in enumerate(class_probabilities)
                    ),
                    key=lambda x: x["class_prob"],
                    reverse=True,
                )
            )
            for class_probabilities in class_probabilities_per_target
        ]

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
        "--logging",
        type=str,
        default="ERROR",
    )

    # if own_args == None -> parse_args will use sys.argv
    # if own_args == [] -> parse_args will use this empty list instead
    own_args = None
    if override_args:
        own_args = []

    # create arguments
    opt = parser.parse_args(args=own_args)

    return opt
