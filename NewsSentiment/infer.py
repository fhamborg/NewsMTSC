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
        self, opt=None, single_targets=True, logging_level="ERROR",
    ):
        self.logger = get_logger()

        default_opts = parse_arguments_from_train(
            override_args=False, overwrite_logging_level=logging_level
        )
        if not opt:
            opt = parse_arguments(override_args=True)

        for key, val in vars(opt).items():
            if val is not None:
                previous_val = getattr(default_opts, key)
                self.logger.info("overwriting: %s=%s to %s", key, previous_val, val)
                setattr(default_opts, key, val)

        default_opts.single_targets = single_targets
        default_opts.multi_targets = not single_targets

        # set training_mode to False so that we get the Instructor object
        default_opts.training_mode = False

        # prepare and initialize instructor
        instructor = prepare_and_start_instructor(default_opts)

        # get stuff that we need from the instructor
        self.model = instructor.own_model
        self.tokenizer = instructor.own_tokenizer
        self.opt = instructor.opt
        self.instructor = instructor
        self.polarities_inverse = SentimentClasses.get_polarity_associations_inverse()

        # set model to evaluation mode (disable gradient / learning)
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

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
        assert not text_left and text or text_left and not text

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


if __name__ == "__main__":
    """
    these are just some tests to understand how to invoke the classifier
    TODO here we should have an argparser and functionality so that the tool can also
    be invoked from cli
    """
    opt = parse_arguments(override_args=False)

    tsc = TargetSentimentClassifier(opt)
    # print(
    #     tsc.infer(
    #         text_left="Mr. Smith said that ",
    #         target_mention="John",
    #         text_right=" was a liar.",
    #     )[0]
    # )
    print(
        tsc.infer(
            text_left="Whatever you think of ",
            target_mention="President Trump",
            text_right=", you have to admit that he’s an astute reader of politics.",
        )[0]
    )
    # print(
    #     tsc.infer(
    #         text="A former employee of the Senate intelligence committee, James A. "
    #              "Wolfe, has been arrested on charges of lying to the FBI about "
    #              "contacts with multiple reporters and appeared in federal court "
    #              "Friday in Baltimore.",
    #         target_mention_from=56,
    #         target_mention_to=70,
    #     )[0]
    #

    # with jsonlines.open("experiments/test.jsonl", "r",) as reader:
    #     lines = [line for line in reader]
    #     # lines = lines[:2]
    #     classifications = []
    #     for line in tqdm(lines):
    #         sentence_normalized = line["sentence_normalized"]
    #         primary_gid = line["primary_gid"]
    #         for target in line["targets"]:
    #             classification = tsc.infer(
    #                 text=sentence_normalized,
    #                 target_mention_from=target["from"],
    #                 target_mention_to=target["to"],
    #             )
    #             result = {
    #                 "primary_gid": primary_gid,
    #                 "sentence": sentence_normalized,
    #                 "target": target["mention"],
    #                 "from": target["from"],
    #                 "to": target["to"],
    #                 "positive": tsc.get_info_for_label(classification, "positive")[
    #                     "class_prob"
    #                 ],
    #                 "neutral": tsc.get_info_for_label(classification, "neutral")[
    #                     "class_prob"
    #                 ],
    #                 "negative": tsc.get_info_for_label(classification, "negative")[
    #                     "class_prob"
    #                 ],
    #                 "y_pred": classification[0]["class_label"],
    #                 "y_true": SentimentClasses.polarity2label(target["polarity"]),
    #                 "is_correct": classification[0]["class_label"]
    #                 == SentimentClasses.polarity2label(target["polarity"]),
    #             }
    #             classifications.append(result)
    #     import pandas as pd
    #
    #     df = pd.DataFrame(classifications)
    #     df.to_excel("results.xlsx")
