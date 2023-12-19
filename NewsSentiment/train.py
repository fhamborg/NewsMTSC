import argparse
import math
import os
import random
import sys
import time
from collections import Counter
from typing import Iterable
import logging

import numpy
import torch
import torch.nn as nn
from jsonlines import jsonlines
from torch.utils.data import DataLoader
from transformers import (
    BertModel,
    XLNetModel,
    RobertaModel,
    PreTrainedModel,
    BertTokenizer,
    RobertaTokenizer,
    XLNetTokenizer,
    AlbertTokenizer,
    AlbertModel,
    PreTrainedTokenizer,
)

from NewsSentiment.SentimentClasses import SentimentClasses
from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset, RandomOversampler, FXEasyTokenizer
from NewsSentiment.download import Download
from NewsSentiment.earlystopping import EarlyStopping
from NewsSentiment.evaluator import Evaluator
from NewsSentiment.fxlogger import get_logger
from NewsSentiment.knowledge.bingliuopinion.bingliuopinion import (
    get_num_bingliu_polarities,
)
from NewsSentiment.knowledge.mpqasubjectivity.mpqasubjectivity import (
    get_num_mpqa_subjectivity_polarities,
)
from NewsSentiment.knowledge.nrcemolex.nrcemolex import get_num_nrc_emotions
from NewsSentiment.knowledge.zeros.zerosknowledge import get_num_zero_dimensions
from NewsSentiment.losses.crossentropycrossweight import CrossEntropyLoss_CrossWeight
from NewsSentiment.losses.crossentropylosslsr import CrossEntropyLoss_LSR
from NewsSentiment.losses.crossentropylosswithconfidence import (
    CrossEntropyLossWithConfidence,
)
from NewsSentiment.losses.seq2seqloss import SequenceLoss
from NewsSentiment.models.FXEnsemble import FXEnsemble
from NewsSentiment.models.multitargets.contrasting import Contrasting
from NewsSentiment.models.multitargets.random_multi import RandomMulti
from NewsSentiment.models.multitargets.seq2seq import SeqTwoSeq
from NewsSentiment.models.multitargets.seq2seq_without_targetmask import (
    SeqTwoSeqWithoutTargetMask,
)
from NewsSentiment.models.multitargets.tdbertlikemultitarget import (
    TDBertLikeMultiTarget,
)
from NewsSentiment.models.multitargets.tdbertlikemultitarget_dense import (
    TDBertLikeMultiTargetDense,
)
from NewsSentiment.models.singletarget.aen import AEN_Base
from NewsSentiment.models.singletarget.grutscsingle import GRUTSCSingle
from NewsSentiment.models.singletarget.lcf import LCF_BERT
from NewsSentiment.models.singletarget.lcf2 import LCF_BERT2Dual
from NewsSentiment.models.singletarget.lcfs import LCFS_BERT
from NewsSentiment.models.singletarget.lcfst import LCFST_BERT
from NewsSentiment.models.singletarget.lcft import LCFT_BERT
from NewsSentiment.models.singletarget.notargetcls import NoTargetClsBert
from NewsSentiment.models.singletarget.random_single import RandomSingle
from NewsSentiment.models.singletarget.spc import SPC_Base
from NewsSentiment.models.singletarget.td_bert import TD_BERT
from NewsSentiment.models.singletarget.td_bert_qa import TD_BERT_QA_MUL, TD_BERT_QA_CON
from NewsSentiment.models.singletarget.tdbertlikesingle import TDBertLikeSingle
from NewsSentiment.plotter_utils import create_save_plotted_confusion_matrix

logger = get_logger()

TRANSFORMER_MODELS_INFO = {
    BERT_BASE_UNCASED: {"tokenizer_class": BertTokenizer, "model_class": BertModel},
    ROBERTA_BASE: {"tokenizer_class": RobertaTokenizer, "model_class": RobertaModel},
    XLNET_BASE_CASED: {"tokenizer_class": XLNetTokenizer, "model_class": XLNetModel},
    ALBERT_XXLARGE: {"tokenizer_class": AlbertTokenizer, "model_class": AlbertModel},
    ALBERT_BASE: {"tokenizer_class": AlbertTokenizer, "model_class": AlbertModel},
}

OWN_MODELNAME2CLASS = {
    # class output
    "lcf_bert": LCF_BERT,
    "lcf_bert2": LCF_BERT2Dual,
    "lcfs_bert": LCFS_BERT,
    "lcft_bert": LCFT_BERT,
    "lcfst_bert": LCFST_BERT,
    "spc_bert": SPC_Base,
    "aen_bert": AEN_Base,
    "tdbert": TD_BERT,
    "tdbert-qa-mul": TD_BERT_QA_MUL,
    "tdbert-qa-con": TD_BERT_QA_CON,
    "tdbertlikesingle": TDBertLikeSingle,
    "tdbertlikemulti": TDBertLikeMultiTarget,
    "tdbertlikemulti_dense": TDBertLikeMultiTargetDense,
    "contrasting": Contrasting,
    "grutsc": GRUTSCSingle,
    # sequence output
    "seq2seq": SeqTwoSeq,
    "seq2seq_withouttargetmask": SeqTwoSeqWithoutTargetMask,
    # other baselines
    "notargetclsbert": NoTargetClsBert,
    "random_multi": RandomMulti,
    "random_single": RandomSingle,
    # misc
    "fxensemble": FXEnsemble,
}


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        self.transformer_tokenizers = {}
        self.transformer_models = {}

        own_model_class = OWN_MODELNAME2CLASS[opt.own_model_name]

        # create transformer models
        for pretrained_weights_name in own_model_class.get_language_models():
            self.create_transformer_model(pretrained_weights_name)
        logger.info("initialized transformer tokenizers and models")

        # setup own tokenizer
        self.own_tokenizer = FXEasyTokenizer(
            self.transformer_tokenizers,
            self.opt.max_seq_len,
            self.opt.knowledgesources,
            self.opt.is_use_natural_target_phrase_for_spc,
        )

        self.trainset = None
        self.devset = None
        self.testset = None
        self.all_datasets = None

        if self.opt.training_mode:
            self.load_datasets()

        # get model config (currently supports only a single language model)
        assert len(self.transformer_models) == 1
        transformer_model_config = list(self.transformer_models.values())[0].config
        assert len(self.transformer_tokenizers) == 1
        transformer_tokenizer = list(self.transformer_tokenizers.values())[0]

        # setup own model
        own_model_object = own_model_class(
            transformer_models=self.transformer_models,
            opt=self.opt,
            config=transformer_model_config,
        )
        own_model_object = own_model_object.to(self.opt.device)
        if self.opt.state_dict:
            Download.download(own_model_class)
            logger.info("loading weights from %s...", self.opt.state_dict)
            state_dict = torch.load(self.opt.state_dict, map_location=self.opt.device)
            own_model_object.load_state_dict(state_dict)
            logger.info("done")
        self.own_model = own_model_object
        logger.info("initialized own model")

        if self.opt.export_finetuned_model:
            self.save_pretrained_model(
                own_model_object, transformer_tokenizer, self.opt.export_finetuned_model
            )

        self.evaluator = Evaluator(
            SentimentClasses.get_sorted_expected_label_values(),
            SentimentClasses.get_polarity_associations(),
            self.opt.snem,
            self.opt.is_return_confidence,
        )

        self._print_args()

    def _load_dataset(self, path, coref_mode):
        return FXDataset(
            path,
            self.opt.data_format,
            self.own_tokenizer,
            SentimentClasses.get_polarity_associations(),
            SentimentClasses.get_polarity_associations_inverse(),
            SentimentClasses.get_sorted_expected_label_names(),
            self.opt.single_targets,
            coref_mode,
            self.opt.devmode,
            self.opt.ignore_parsing_errors,
        )

    def load_datasets(self):
        logger.info(
            "loading datasets {} from {}".format(
                self.opt.dataset_name, self.opt.dataset_path
            )
        )
        self.trainset = self._load_dataset(
            self.opt.dataset_path + "train.jsonl",
            coref_mode=self.opt.coref_mode_in_training,
        )
        self.devset = self._load_dataset(
            self.opt.dataset_path + "dev.jsonl", coref_mode="ignore"
        )
        self.testset = self._load_dataset(
            self.opt.dataset_path + "test.jsonl", coref_mode="ignore"
        )
        self.all_datasets = [self.trainset, self.devset, self.testset]
        logger.info("loaded datasets from {}".format(self.opt.dataset_path))

        all_datasets_overflow_counter = Counter()
        for dataset in self.all_datasets:
            all_datasets_overflow_counter += dataset.count_sequence_overflow
        logger.info(
            f"count truncated sequences in total: {all_datasets_overflow_counter}"
        )

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.own_model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            "n_trainable_params: {0}, n_nontrainable_params: {1}".format(
                n_trainable_params, n_nontrainable_params
            )
        )
        logger.info("> training arguments:")
        for arg in vars(self.opt):
            logger.info(">>> {0}: {1}".format(arg, getattr(self.opt, arg)))

    def create_transformer_model(
        self, pretrained_weights_name: str, only_model: bool = False
    ):
        logger.info(
            "creating model for weights name: {}".format(pretrained_weights_name)
        )

        if not only_model:
            model_info = TRANSFORMER_MODELS_INFO[pretrained_weights_name]
            tokenizer_class = model_info["tokenizer_class"]
            model_class = model_info["model_class"]

            if (
                self.opt.pretrained_model_name
                and self.opt.pretrained_model_name is not None
                and self.opt.pretrained_model_name != "default"
            ):
                model_path = self.opt.pretrained_model_name
            else:
                model_path = pretrained_weights_name
            logger.info("using model_path: %s", model_path)

            self.transformer_tokenizers[
                pretrained_weights_name
            ] = tokenizer_class.from_pretrained(model_path)
            
            # supress the transformers warning
            # "Some weights of the model checkpoint..were not used.."
            transformers_logger = logging.getLogger('transformers.modeling_utils')
            transformers_logger_level = transformers_logger.getEffectiveLevel()
            transformers_logger.setLevel(logging.ERROR)

            self.transformer_models[
                pretrained_weights_name
            ] = model_class.from_pretrained(model_path, output_hidden_states=True)
            
            # reset transformers logging level
            transformers_logger.setLevel(transformers_logger_level)

    def _reset_params_of_own_model(self):
        for child in self.own_model.children():
            if not issubclass(
                child.__class__, PreTrainedModel
            ):  # if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1.0 / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _create_prepare_model_path_to_store_state_dict(self, snem, epoch):
        selected_model_filename = "{0}_{1}_val_{2}_{3}_epoch{4}".format(
            self.opt.own_model_name,
            self.opt.dataset_name,
            self.opt.snem,
            round(snem, 4),
            epoch,
        )

        pathdir = os.path.join(self.opt.experiment_path, "state_dict")

        os.makedirs(pathdir, exist_ok=True)
        selected_model_path = os.path.join(pathdir, selected_model_filename)

        return selected_model_filename, selected_model_path

    def select_inputs(self, sample_batched, is_single_item=False):
        """
        The input to this function is a batch of the dataset, aggregated by pytorch's
        DataLoader. The DataLoader performs complex aggregation of multiple items of the
        batch into one vector, e.g.:
        - a field containing a single number or boolean in each item will become a tensor of size
          k, where k is the number of items in the batch. Examples: polarity
        - a field containing a vector (list of) numbers of some size up to m, will
          become a tensor of size (k, m). Examples: target_mask, text_ids_with_special_tokens
        - a field containing a str, will become a list of str of size k. Examples: label

        The function at hand then selects only those fields that are one of
         [FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS, FIELD_TARGET_MASK], for each LM tokenizer
         defined in the FXEasyTokenizer (this information is setup during initializing
          the instructor), specifically in self.own_tokenizer.tokenizers_name_and_obj.
        """
        inputs = []
        params2index = {}
        index = 0
        for weights_name, field_name in self.own_model.get_input_field_ids():
            seq = sample_batched[weights_name][field_name]
            if is_single_item:
                # if we only have a single item, we need to add a virtual, singleton
                # batch dimension
                seq = seq.unsqueeze(0)
            # move to device and append to list
            inputs.append(seq.to(self.opt.device))
            # store in param dict for easy access later
            param_id = (weights_name, field_name)
            params2index[param_id] = index
            index += 1

        FXDataset.set_params2index(params2index)

        return inputs

    def _train(self, criterion, optimizer, train_data_loader, dev_data_loader):
        global_step = 0
        selected_model_path = None
        selected_model_filename = None
        selected_model_dev_stats = None

        # initialize the early_stopping object
        early_stopping = EarlyStopping()

        # skip training
        if self._is_random_model():
            logger.info(
                "skipping training because random model: %s", self.opt.own_model_name
            )
            # perform one fake intraining evaluation so that we can save the model and
            # use that for testset evaluation
            # set epoch to a bit higher than max number of epochs to ensure that the
            # model is being saved
            fake_epoch = self.opt.num_epoch + 2
            (
                selected_model_path,
                selected_model_filename,
                selected_model_dev_stats,
            ) = self._intraining_evaluation_and_model_save(
                dev_data_loader,
                early_stopping,
                fake_epoch,
                selected_model_path,
                selected_model_filename,
                selected_model_dev_stats,
            )
        else:
            for epoch in range(self.opt.num_epoch):
                logger.info(">" * 100)
                logger.info(
                    "epoch: {} (num_epoch: {})".format(epoch, self.opt.num_epoch)
                )
                num_correct_predictions_total = 0
                num_predictions_total = 0
                num_examples_total = 0
                loss_total = 0

                # switch model to training mode
                self.own_model.train()

                # train on batches
                for i_batch, sample_batched in enumerate(train_data_loader):
                    global_step += 1
                    # clear gradient accumulators
                    optimizer.zero_grad(set_to_none = False)
                    # select only relevant fields
                    inputs = self.select_inputs(sample_batched)
                    targets = sample_batched["polarity"].to(self.opt.device)

                    # invoke the model
                    outputs = self.own_model(inputs)

                    # some models return multiple objects
                    if self.opt.own_model_name in ["contrasting"]:
                        outputs, cross_weight = outputs

                    # since the following logic is based on multi targets, but in case of
                    # single_targets=True this dimension is removed (since the state of the
                    # art models for single target per item, such as LCF, were not adapted
                    # to multiple targets, and thus output only a two dimensional tensor
                    # (batch, polarity)), we need to add a singleton dimension representing
                    # the target dimension
                    if self.opt.single_targets:
                        outputs = outputs.unsqueeze(1)

                    # permute to match k-dimensional crossentropyloss,
                    # see https://pytorch.org/docs/stable/nn.html#crossentropyloss
                    outputs = outputs.permute(0, 2, 1)

                    # apply loss
                    if self.opt.loss == "sequence":
                        # sequence loss requires targets mask (indev: currently, this will
                        # work only for the "seq2seq" model)
                        text_bert_indices_targets_mask = FXDataset.get_input_by_params(
                            inputs,
                            get_default_lm(),
                            FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK,
                        )
                        loss = criterion(
                            outputs, targets, text_bert_indices_targets_mask
                        )
                    elif self.opt.loss == "crossentropy_crossweight":
                        text_bert_indices_targets_mask = FXDataset.get_input_by_params(
                            inputs,
                            get_default_lm(),
                            FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK,
                        )
                        loss = criterion(
                            outputs,
                            targets,
                            cross_weight,
                            text_bert_indices_targets_mask,
                        )
                    elif self.opt.is_return_confidence:
                        loss = criterion(outputs, targets)
                    else:
                        loss = criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()

                    # calculate training stats
                    # get predicted classes
                    if self.opt.loss == "sequence":
                        predicted_classes = self._get_classes_from_sequence_output(
                            outputs, text_bert_indices_targets_mask
                        )
                    else:
                        predicted_classes = torch.argmax(outputs, dim=1)

                    # calculate number of correctly predicted classes and sum it. here, we
                    # can ignore SentimentClasses.FILLUP_POLARITY_VALUE because when in
                    # a cell in "targets" the value is
                    # SentimentClasses.FILLUP_POLARITY_VALUE, the corresponding value in
                    # predicted_classes will never be the same (so this will never attribute
                    # to a correct prediction)
                    num_correct_predictions = (
                        (predicted_classes == targets).sum().item()
                    )
                    # to count number of all predictions, we need to ignore those that have
                    # SentimentClasses.FILLUP_POLARITY_VALUE in a cell of "targets"
                    # note, that we do not use the number of examples but the number of
                    # predictions, since one training example can contain one or more
                    # targets
                    num_predictions = targets != SentimentClasses.FILLUP_POLARITY_VALUE
                    num_predictions = num_predictions.sum().item()
                    # ensure the number of predictions in this batch is equal to the number
                    # of targets defined by the dataset.py
                    assert num_predictions == sample_batched["num_targets"].sum().item()
                    # get number of examples (we can use more or less any field in
                    # sample_batched to get the number of examples)
                    num_examples = len(sample_batched["num_targets"])

                    # update stats
                    num_correct_predictions_total += num_correct_predictions
                    num_predictions_total += num_predictions
                    loss_total += loss.item() * num_predictions
                    num_examples_total += num_examples
                    if global_step % self.opt.log_step == 0:
                        # update accuracy
                        train_acc = (
                            num_correct_predictions_total / num_predictions_total
                        )
                        train_loss = loss_total / num_predictions_total
                        logger.info(
                            "loss: {:.4f}, acc: {:.4f}, ex.: {}, pred.: {}/{}".format(
                                train_loss,
                                train_acc,
                                num_examples_total,
                                num_correct_predictions_total,
                                num_predictions_total,
                            )
                        )

                # completed all batches of current epoch. now, perform a devset
                # evaluation and if best model so far, save it
                (
                    selected_model_path,
                    selected_model_filename,
                    selected_model_dev_stats,
                ) = self._intraining_evaluation_and_model_save(
                    dev_data_loader,
                    early_stopping,
                    epoch,
                    selected_model_path,
                    selected_model_filename,
                    selected_model_dev_stats,
                )

                if early_stopping.early_stop and self.opt.use_early_stopping:
                    logger.info(
                        "early stopping after {} epochs without improvement, total epochs: {} of {}".format(
                            early_stopping.patience, epoch, self.opt.num_epoch
                        )
                    )
                    break

        return selected_model_path, selected_model_filename, selected_model_dev_stats

    def _intraining_evaluation_and_model_save(
        self,
        dev_data_loader,
        early_stopping,
        epoch,
        selected_model_path,
        selected_model_filename,
        selected_model_dev_stats,
    ):
        dev_stats = self._evaluate(dev_data_loader)
        dev_snem = dev_stats[self.opt.snem]
        self.evaluator.print_stats(dev_stats, "dev during training")
        early_stopping(dev_snem)

        (
            has_stored,
            _selected_model_filename,
            _selected_model_path,
            _selected_model_dev_stats,
        ) = self._save_model_state_dict(early_stopping, epoch, dev_stats)
        if has_stored:
            selected_model_dev_stats = _selected_model_dev_stats
            selected_model_path = _selected_model_path
            selected_model_filename = _selected_model_filename

        return selected_model_path, selected_model_filename, selected_model_dev_stats

    def save_pretrained_model(
        self,
        pretrained_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        save_directory: str,
    ):
        """
        Used to create a model that can be uploaded to huggingface hub
        :param finetuned_pretrained_model: Should typically be finetuned already
        :param save_directory:
        :return:
        """
        pretrained_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        tokenizer.save_vocabulary(save_directory)

        logger.info("model exported to: %s", save_directory)

    def _save_model_state_dict(self, early_stopping, epoch: int, dev_stats):
        """
        Used during evaluation to store the best performing model in pytorch format
        :param early_stopping:
        :param epoch:
        :param dev_stats:
        :return:
        """
        dev_snem = dev_stats[self.opt.snem]

        has_stored = False
        selected_model_filename = None
        selected_model_path = None
        selected_model_dev_stats = None

        if self.opt.eval_only_after_last_epoch:
            if epoch >= self.opt.num_epoch - 1:
                # return the model that was trained through all epochs as the selected model
                logger.info("all epochs finished, saving model to disk...")

                has_stored = True
                selected_model_dev_stats = dev_stats

                (
                    selected_model_filename,
                    selected_model_path,
                ) = self._create_prepare_model_path_to_store_state_dict(dev_snem, epoch)
                torch.save(self.own_model.state_dict(), selected_model_path)
                logger.info(">> saved: {}".format(selected_model_path))

                # save confusion matrices
                filepath_stats_base = os.path.join(
                    self.opt.experiment_path, "statistics", selected_model_filename
                )
                if not filepath_stats_base.endswith("/"):
                    filepath_stats_base += "/"
                os.makedirs(filepath_stats_base, exist_ok=True)
                create_save_plotted_confusion_matrix(
                    dev_stats["confusion_matrix"],
                    expected_labels=SentimentClasses.get_sorted_expected_label_values(),
                    basepath=filepath_stats_base,
                )
                logger.debug(
                    "created confusion matrices in path: {}".format(filepath_stats_base)
                )
        else:
            # return the best model during any epoch
            if early_stopping.flag_has_score_increased_since_last_check:
                logger.info("model yields best performance so far, saving to disk...")
                has_stored = True
                selected_model_dev_stats = dev_stats
                (
                    selected_model_filename,
                    selected_model_path,
                ) = self._create_prepare_model_path_to_store_state_dict(dev_snem, epoch)

                torch.save(self.own_model.state_dict(), selected_model_path)
                logger.info(">> saved: {}".format(selected_model_path))

                # save confusion matrices
                filepath_stats_base = os.path.join(
                    self.opt.experiment_path, "statistics", selected_model_filename
                )
                if not filepath_stats_base.endswith("/"):
                    filepath_stats_base += "/"
                os.makedirs(filepath_stats_base, exist_ok=True)
                create_save_plotted_confusion_matrix(
                    dev_stats["confusion_matrix"],
                    expected_labels=SentimentClasses.get_sorted_expected_label_values(),
                    basepath=filepath_stats_base,
                )
                logger.debug(
                    "created confusion matrices in path: {}".format(filepath_stats_base)
                )
        return (
            has_stored,
            selected_model_filename,
            selected_model_path,
            selected_model_dev_stats,
        )

    def _get_classes_from_sequence_output(
        self, model_outputs, text_bert_indices_targets_mask
    ):
        """

        :param model_outputs: Tensor of shape:
            batch, numclasses, seqlen
        :param text_bert_indices_targets_mask: Tensor of shape:
            batch, targets, seqlen
        :return: Tensor of shape:
            batch, targets
        """
        num_targets = text_bert_indices_targets_mask.shape[1]
        num_classes = model_outputs.shape[1]

        # prepare model output
        model_outputs = model_outputs.unsqueeze(1)
        model_outputs = model_outputs.repeat(1, num_targets, 1, 1)
        # new shape: batch, targets, numclasses, seqlen

        # prepare text_bert_indices_targets_mask
        text_bert_indices_targets_mask = text_bert_indices_targets_mask.unsqueeze(2)
        text_bert_indices_targets_mask = text_bert_indices_targets_mask.repeat(
            1, 1, num_classes, 1
        )
        # new shape: batch, targets, numclasses, seqlen

        # multiply so that we will only have the values model output values that belong
        # to a target node (1's for output nodes with a target in text_bert_indices...
        # 0's for non-targets)
        only_target_outputs = model_outputs * text_bert_indices_targets_mask
        # shape: batch, targets, numclasses, seqlen

        # aggregate, e.g., mean
        # note that in principle any way of reducing (or aggregating) could be used here
        # e.g., also sum or max. however, the reducing method should as good as possible
        # match the way of calculating the sequence based loss (seq2seqloss.py). our
        # seq-based loss currently equally assigns the loss to each target output node
        # so that in my opinion the unweighted mean is most appropriate here for
        # reduction, since it also equally uses the output of each target node
        only_target_outputs_aggregated = only_target_outputs.mean(3)
        # shape: batch, targets, classes

        predicted_classes = torch.argmax(only_target_outputs_aggregated, dim=2)
        # predicted_classes shape: batch, targets
        return predicted_classes

    def _evaluate(self, data_loader, get_examples=False, basepath=None):
        t_labels_all = None
        t_outputs_all = None
        t_text_bert_indices_targets_mask_all = None
        t_texts_all = []
        t_outputs_confidence = None

        # switch model to evaluation mode
        self.own_model.eval()

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = self.select_inputs(t_sample_batched)
                t_labels = t_sample_batched["polarity"].to(self.opt.device)
                t_text_bert_indices_targets_mask = FXDataset.get_input_by_params(
                    t_inputs,
                    self.own_model.get_language_models()[0],
                    FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK,
                )
                assert len(self.own_model.get_language_models()) == 1, (
                    f"multiple language models per own model are currently not supported: "
                    f"{self.own_model.get_language_models()}, "
                    f"{len(self.own_model.get_language_models())}"
                )

                t_texts = t_sample_batched["text"]
                # t_targets = t_sample_batched["orig_targets"]

                # invoke the model
                t_outputs = self.own_model(t_inputs)
                # some models return multiple objects
                if self.opt.own_model_name in ["contrasting"]:
                    t_outputs, t_cross_weight = t_outputs

                if self.opt.single_targets:
                    t_outputs = t_outputs.unsqueeze(1)

                if t_labels_all is None:
                    t_labels_all = t_labels
                    t_outputs_all = t_outputs
                    t_text_bert_indices_targets_mask_all = (
                        t_text_bert_indices_targets_mask
                    )
                else:
                    t_labels_all = torch.cat((t_labels_all, t_labels), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                    t_text_bert_indices_targets_mask_all = torch.cat(
                        (
                            t_text_bert_indices_targets_mask_all,
                            t_text_bert_indices_targets_mask,
                        ),
                        dim=0,
                    )

                t_texts_all.extend(t_texts)
                # t_targets_all.extend(t_targets)

        y_true = t_labels_all.cpu()

        if self.opt.loss == "sequence":
            # we first need to convert sequence-based predictions to target-based
            # predictions
            # permute to match requirement of _get_classes_from_sequence_output
            t_outputs_all = t_outputs_all.permute(0, 2, 1)
            y_pred = self._get_classes_from_sequence_output(
                t_outputs_all, t_text_bert_indices_targets_mask_all
            ).cpu()
        else:
            # softmax: get predictions from outputs
            # have to take the 3rd (dim=2) dimension
            if self.opt.is_return_confidence:
                # we have to remove the last class because this is the confidence and
                # we do not want to evaluate on the confidence (at least not now)
                t_outputs_confidence = t_outputs_all.clone().cpu()[:, :, -1]
                t_outputs_all = t_outputs_all[:, :, :-1]
            y_pred = torch.argmax(t_outputs_all, dim=2).cpu()

        stats = self.evaluator.calc_statistics(y_true, y_pred, t_outputs_confidence)

        if get_examples:
            self.evaluator.write_error_table(
                y_true, y_pred, t_texts_all, basepath + "errortable.jsonl",
            )

        return stats

    def get_normalized_inv_class_frequencies(self):
        inv_freqs = []
        for label_name in self.sorted_expected_label_names:
            inv_freq_of_class = 1.0 / self.testset.label_counter[label_name]
            inv_freqs.append(inv_freq_of_class)

        sum_of_inv_freqs = sum(inv_freqs)
        for i in range(len(inv_freqs)):
            inv_freqs[i] = inv_freqs[i] / sum_of_inv_freqs

        return inv_freqs

    def _is_random_model(self):
        return self.opt.own_model_name in ["random_multi", "random_single"]

    def run(self):
        # balancing modes
        class_weights = None
        sampler_train = None
        assert self.opt.balancing is None, (
            "balancing is currently only implemented for None. if we want to have it, "
            "we need to come up with a logic that considers that an item can have "
            "multiple polarities in mtsc-mode"
        )
        if self.opt.balancing == "lossweighting":
            inv_class_freqs = self.get_normalized_inv_class_frequencies()
            logger.info("weighting losses of classes: {}".format(inv_class_freqs))
            class_weights = torch.tensor(inv_class_freqs).to(self.opt.device)
        elif self.opt.balancing == "oversampling":
            sampler_train = RandomOversampler(self.trainset, self.opt.seed)

        if self.opt.loss == "crossentropy":
            if self.opt.is_return_confidence:
                criterion = CrossEntropyLossWithConfidence(
                    weight=class_weights,
                    ignore_index=SentimentClasses.FILLUP_POLARITY_VALUE,
                )
            else:
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights,
                    ignore_index=SentimentClasses.FILLUP_POLARITY_VALUE,
                )
        elif self.opt.loss == "crossentropy_lsr":
            criterion = CrossEntropyLoss_LSR(
                self.opt.device, smoothing_value=0.2, weight=class_weights
            )
        elif self.opt.loss == "crossentropy_crossweight":
            criterion = CrossEntropyLoss_CrossWeight(
                device=self.opt.device,
                weight=class_weights,
                ignore_index=SentimentClasses.FILLUP_POLARITY_VALUE,
            )
        elif self.opt.loss == "sequence":
            criterion = SequenceLoss(device=self.opt.device, weight=class_weights)
        else:
            raise ValueError("loss unknown: {}".format(self.opt.loss))

        # optimizer
        if self._is_random_model():
            logger.info("initialize optimizer")
            optimizer = None
        else:
            _params = filter(lambda p: p.requires_grad, self.own_model.parameters())
            optimizer = self.opt.optimizer(
                _params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg
            )

        # data loaders
        if sampler_train:
            train_data_loader = DataLoader(
                dataset=self.trainset,
                batch_size=self.opt.batch_size,
                sampler=sampler_train,
            )
        else:
            train_data_loader = DataLoader(
                dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True
            )
        dev_data_loader = DataLoader(
            dataset=self.devset, batch_size=self.opt.batch_size, shuffle=False
        )
        test_data_loader = DataLoader(
            dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False
        )

        # start training
        self._reset_params_of_own_model()
        logger.info("starting training...")
        time_training_start = time.time()
        best_model_path, best_model_filename, selected_model_dev_stats = self._train(
            criterion, optimizer, train_data_loader, dev_data_loader
        )
        time_training_elapsed_mins = (time.time() - time_training_start) // 60
        logger.info(
            "training finished. duration [mins]: {}".format(time_training_elapsed_mins)
        )

        self.perform_post_training_actions(
            best_model_path,
            best_model_filename,
            test_data_loader,
            selected_model_dev_stats,
            time_training_elapsed_mins,
        )

    def get_serializable_stats(self, stats):
        sstats = stats.copy()
        sstats["recalls_of_classes"] = stats["recalls_of_classes"].tolist()
        sstats["confusion_matrix"] = stats["confusion_matrix"].tolist()
        return sstats

    def get_serializable_opts(self):
        opts = vars(self.opt)
        sopts = opts.copy()
        del sopts["optimizer"]
        del sopts["initializer"]
        del sopts["device"]
        return sopts

    def perform_post_training_actions(
        self,
        selected_model_path,
        selected_model_filename,
        test_data_loader,
        selected_model_dev_stats,
        time_training_elapsed_mins,
    ):
        logger.info(
            "loading selected model from training: {}".format(selected_model_path)
        )
        self.own_model.load_state_dict(torch.load(selected_model_path))

        logger.info("evaluating selected model on test-set")
        # set model into evaluation mode (cf. https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train)
        self.own_model.eval()

        # do the actual evaluation
        filepath_stats_prefix = os.path.join(
            self.opt.experiment_path, "statistics", selected_model_filename
        )
        os.makedirs(filepath_stats_prefix, exist_ok=True)
        if not filepath_stats_prefix.endswith("/"):
            filepath_stats_prefix += "/"

        test_stats = self._evaluate(
            test_data_loader, get_examples=True, basepath=filepath_stats_prefix
        )
        test_snem = test_stats[self.opt.snem]

        self.evaluator.print_stats(test_stats, "evaluation on test-set")

        # save dev and test results
        experiment_results = {}
        experiment_results["test_stats"] = self.get_serializable_stats(test_stats)
        experiment_results["dev_stats"] = self.get_serializable_stats(
            selected_model_dev_stats
        )
        experiment_results["options"] = self.get_serializable_opts()
        experiment_results["time_training_elapsed_mins"] = time_training_elapsed_mins

        experiment_results_path = os.path.join(
            self.opt.experiment_path, "experiment_results.jsonl"
        )
        with jsonlines.open(experiment_results_path, "w") as writer:
            writer.write(experiment_results)

        # save confusion matrices
        test_confusion_matrix = test_stats["confusion_matrix"]

        create_save_plotted_confusion_matrix(
            test_confusion_matrix,
            expected_labels=SentimentClasses.get_sorted_expected_label_values(),
            basepath=filepath_stats_prefix,
        )

        logger.info("finished execution of this run. exiting.")

        # print snem pad_value to stdout, for the controller to parse it
        print(test_snem)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean pad_value expected.")


def _setup_cuda(opt):
    logger.info("cuda information")
    logger.info("scc SGE_GPU: %s", os.environ.get("SGE_GPU"))
    logger.info("arg: cuda device: %s", opt.device)

    if torch.cuda.is_available():
        logger.info("cuda is available on this machine")
        if opt.device == "cpu":
            logger.info("forced use of cpu")
            opt.device = torch.device("cpu")
        elif opt.device is None:
            logger.info("no cuda device defined by argument, using default cuda:0")
            opt.device = torch.device("cuda:0")
        else:
            assert opt.device and opt.device != "cpu"
            logger.info("using cuda device defined by argument")
            opt.device = torch.device(opt.device)
    else:
        opt.device = torch.device("cpu")

    if opt.device.type == "cuda":
        logger.info("GPGPU enabled. CUDA dev index: %s", opt.device.index)
        logger.info(
            "using GPU (cuda memory allocated: %s)", torch.cuda.memory_allocated()
        )
    else:
        logger.info("using CPU")


def _die_gracefully(reason: str):
    logger.info("grateful exit:")
    logger.info(reason)
    # traceback.print_stack()
    sys.exit(99)


def check_arguments(opt):
    """
    Perform simple check of arguments passed to the tool. Idea is to die gratefully as
    early as possible with a special RC = -99 if conditions are not met, e.g., neither
    single_targets nor multi_targets mode is enabled, or if single_targets mode is used
    with a loss function for multi targets, etc.

    :param opt:
    :return:
    """

    if not (opt.single_targets != opt.multi_targets):
        _die_gracefully("you must use either " "single_targets or multi_targets")

    if opt.eval_only_after_last_epoch:
        if opt.use_early_stopping:
            _die_gracefully(
                "early stopping and eval only after last epoch both enabled"
            )

    # loss
    # some easy validation
    if opt.own_model_name in ["seq2seq", "seq2seq_withouttargetmask"]:
        if opt.loss != "sequence":
            _die_gracefully(f"model_name: {opt.own_model_name}, loss: {opt.loss}")
    else:
        if opt.loss == "sequence":
            _die_gracefully(f"model_name: {opt.own_model_name}, loss: {opt.loss}")

    if opt.own_model_name in ["contrasting"]:
        if opt.loss != "crossentropy_crossweight":
            _die_gracefully(f"model_name: {opt.own_model_name}, loss: {opt.loss}")
    else:
        if opt.loss == "crossentropy_crossweight":
            _die_gracefully(f"model_name: {opt.own_model_name}, loss: {opt.loss}")

    # check knowledge source
    assert type(opt.knowledgesources) == tuple, "opt.knowledgesources is not tuple"
    for source in opt.knowledgesources:
        if source not in [
            "bingliu_opinion",
            "mpqa_subjectivity",
            "nrc_emotions",
            "liwc",
            "zeros",
        ]:
            _die_gracefully(f"incorrect source: {source} (of '{opt.knowledgesources}')")

    # coref mode
    # single-target mode: one target per example
    # coref ignore: as before
    #       in_targetmask: just add to each target's targetmask (more 1's in there)
    #       additional_ex: create additional examples
    # multi-target mode: k targets per example
    # coref ignore: as before
    #       in_targetmask: see above
    #       additional_ex: create additional examples
    # but all of this ONLY for train set!
    assert opt.coref_mode_in_training in [
        "ignore",
        "in_targetmask",
        "additional_examples",
    ]


def post_process_arguments(opt):
    # if neither single nor multi targets is enabled, assume default, i.e., single
    if not opt.single_targets and not opt.multi_targets:
        opt.single_targets = True

    # post process
    if type(opt.knowledgesources) == str:
        # should be a list but sometimes this does not work...
        opt.knowledgesources = opt.knowledgesources.split(" ")
    elif isinstance(opt.knowledgesources, Iterable) and len(opt.knowledgesources) == 1:
        # actually, if multiple knowledge sources are passed by controller, they will be
        # treated as a quoted str so that a list of 1 item is created
        opt.knowledgesources = opt.knowledgesources[0].split(" ")
    opt.knowledgesources = tuple(opt.knowledgesources)

    for eks in opt.knowledgesources:
        if eks == "nrc_emotions":
            num_categories = get_num_nrc_emotions()
        elif eks == "mpqa_subjectivity":
            num_categories = get_num_mpqa_subjectivity_polarities()
        elif eks == "bingliu_opinion":
            num_categories = get_num_bingliu_polarities()
        elif eks == "liwc":
            from knowledge.liwc.liwc import get_num_liwc_categories

            num_categories = get_num_liwc_categories()
        elif eks == "zeros":
            num_categories = get_num_zero_dimensions()
        else:
            raise NotImplementedError(f"unknown knowledgesource: {eks}")
        FXEasyTokenizer.NUM_CATEGORIES_OF_SELECTED_KNOWLEDGE_SOURCES += num_categories
        logger.info(
            "updated total number of categories to %s with EKS %s",
            FXEasyTokenizer.NUM_CATEGORIES_OF_SELECTED_KNOWLEDGE_SOURCES,
            eks,
        )


def prepare_and_start_instructor(opt):
    # overwrite default lm
    set_default_lm(opt.default_lm)
    logger.info("set default language model to %s", get_default_lm())

    post_process_arguments(opt)

    # check arguments
    check_arguments(opt)

    if opt.targetclasses == "newsmtsc3":
        SentimentClasses.Sentiment3ForNewsMtsc()
    elif opt.targetclasses == "newsmtsc3strong":
        SentimentClasses.SentimentStrong3ForNewsMtsc()
    elif opt.targetclasses == "newsmtsc3weak":
        SentimentClasses.SentimentWeak3ForNewsMtsc()
    else:
        raise ValueError("")
    assert (
        opt.polarities_dim == -1
    ), "polarities_dim needs to be -1 for automatic detection"
    opt.polarities_dim = len(SentimentClasses.SENTIMENT_CLASSES)
    logger.info("set number of polarity classes to %s", opt.polarities_dim)

    if not opt.balancing or opt.balancing == "None":
        opt.balancing = None

    if opt.seed is None:
        opt.seed = int(time.time())
        logger.info("no random seed was given, using system time")
    logger.info("setting random seed: {}".format(opt.seed))
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    initializers = {
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal,
        "orthogonal_": torch.nn.init.orthogonal_,
    }
    optimizers = {
        "adadelta": torch.optim.Adadelta,  # default lr=1.0
        "adagrad": torch.optim.Adagrad,  # default lr=0.01
        "adam": torch.optim.Adam,  # default lr=0.001
        "adamax": torch.optim.Adamax,  # default lr=0.002
        "asgd": torch.optim.ASGD,  # default lr=0.01
        "rmsprop": torch.optim.RMSprop,  # default lr=0.01
        "sgd": torch.optim.SGD,
    }

    # support for custom pretrained language models, e.g., domain adapted models
    if not opt.pretrained_model_name or opt.pretrained_model_name == "default":
        opt.pretrained_model_name = None
    else:
        opt.pretrained_model_name = os.path.join(
            opt.base_path, "pretrained_models", opt.pretrained_model_name
        )

    # statedicts
    if opt.state_dict is not None:
        if os.path.isfile(opt.state_dict) or opt.state_dict == "pretrained":
            # if exists or using pretrained, nothing to do
            pass
        else:
            # if not exists, prefix default path
            new_path = os.path.join("pretrained_models", "state_dicts", opt.state_dict)
            opt.state_dict = new_path
    else:
        # if not defined, nothing to do
        pass

    if not opt.experiment_path:
        opt.experiment_path = "."
    if not opt.experiment_path.endswith("/"):
        opt.experiment_path = opt.experiment_path + "/"

    _setup_cuda(opt)

    if opt.training_mode:
        logger.debug("dataset_path not defined, creating from dataset_name...")
        opt.dataset_path = os.path.join("datasets", opt.dataset_name)
        if not opt.dataset_path.endswith("/"):
            opt.dataset_path = opt.dataset_path + "/"
        logger.debug(
            "dataset_path created from dataset_name: {}".format(opt.dataset_path)
        )

        # set dataset_path to include experiment_path
        opt.dataset_path = os.path.join(opt.experiment_path, opt.dataset_path)

    ins = Instructor(opt)

    if opt.training_mode:
        opt.initializer = initializers[opt.initializer]
        opt.optimizer = optimizers[opt.optimizer]

        ins.run()
    else:
        return ins


def parse_arguments(override_args=False, overwrite_logging_level=None):
    """
    Where applicable, for each argument a default is defined that led to the best
    results in our experiments for grutsc (which itself is set as a default for the
    corresponding argument).
    :param override_args:
    :param overwrite_logging_level:
    :return:
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_mode", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument("--own_model_name", type=str)
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="name of the sub-folder in 'datasets' containing files named [train,dev,test].jsonl",
    )
    parser.add_argument("--data_format", default="newsmtsc", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--initializer", default="xavier_uniform_", type=str)
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="try 5e-5, 2e-5 for BERT, 1e-3 for others",
    )
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--l2reg", default=0.01, type=float)
    parser.add_argument(
        "--num_epoch",
        default=3,
        type=int,
        help="try larger number for non-BERT models",
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="try 16, 32, 64 for BERT models"
    )
    parser.add_argument("--log_step", default=5, type=int)
    parser.add_argument("--max_seq_len", default=150, type=int)
    parser.add_argument("--polarities_dim", default=-1, type=int)
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="e.g., cuda:0; if None, CPU will be used",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="set seed for reproducibility"
    )
    parser.add_argument(
        "--local_context_focus",
        default="cdm",
        type=str,
        help="local context focus mode, cdw or cdm",
    )
    parser.add_argument(
        "--SRD",
        default=3,
        type=int,
        help="semantic-relative-distance, see the paper of LCF-BERT " "model",
    )
    parser.add_argument(
        "--snem", default="f1_macro", help="see evaluator.py for valid options"
    )
    parser.add_argument(
        "--devmode",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="devmode, default off, enable by using True",
    )
    parser.add_argument(
        "--experiment_path",
        default="./NewsSentiment/experiments/default",
        type=str,
        help="if defined, all data will be read from / saved to a folder in the experiments folder",
    )
    parser.add_argument("--balancing", type=str, default=None)
    parser.add_argument("--spc_lm_representation", type=str, default="mean_last")
    parser.add_argument(
        "--spc_input_order",
        type=str,
        default="text_target",
        help="SPC: order of input; target_text " "or text_target",
    )
    parser.add_argument(
        "--use_early_stopping", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--eval_only_after_last_epoch",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="if False, evaluate the best model that was seen during any training epoch. if True, "
        "evaluate only the model that was trained through all num_epoch epochs.",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="default",
        help="has to be placed in folder pretrained_models",
    )
    parser.add_argument("--state_dict", type=str, default=None)
    parser.add_argument(
        "--single_targets", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--multi_targets", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument("--loss", type=str, default="crossentropy")
    parser.add_argument("--targetclasses", type=str, default="newsmtsc3")
    parser.add_argument(
        "--knowledgesources",
        nargs="+",
        default=["nrc_emotions", "mpqa_subjectivity", "bingliu_opinion"],
    )
    parser.add_argument(
        "--is_use_natural_target_phrase_for_spc",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--default_lm", type=str, default=ROBERTA_BASE)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--coref_mode_in_training", type=str, default="ignore")
    parser.add_argument(
        "--logging", type=str, default="INFO",
    )
    parser.add_argument(
        "--ignore_parsing_errors", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument("--export_finetuned_model", type=str, default=None)
    parser.add_argument(
        "--is_return_confidence", type=str2bool, nargs="?", const=True, default=False
    )

    # if own_args == None -> parse_args will use sys.argv
    # if own_args == [] -> parse_args will use this empty list instead
    own_args = None
    if override_args:
        own_args = []

    # create arguments
    opt = parser.parse_args(args=own_args)

    # add basepath
    dir_path = os.path.dirname(os.path.realpath(__file__))
    opt.base_path = dir_path

    # set logging
    if overwrite_logging_level:
        logger.setLevel(overwrite_logging_level)
    else:
        logger.setLevel(opt.logging)

    return opt


if __name__ == "__main__":
    logger.info("python bin: %s", sys.executable)
    opt = parse_arguments()
    prepare_and_start_instructor(opt)
