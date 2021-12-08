from NewsSentiment.consts import BERT_BASE_UNCASED

combinations_default_0 = {
    "own_model_name": [
        # baselines: single
        "notargetclsbert",
        "lcf_bert",
        "lcf_bert2",
        "lcfs_bert",
        "lcft_bert",
        "aen_bert",
        "spc_bert",
        "tdbert",
        "tdbert-qa-mul",
        "tdbert-qa-con",
        # own models: single
        "tdbertlikesingle",
        "lcfst_bert",
        "grutsc",
        # own models: multi
        "tdbertlikemulti",
        # "tdbertlikemulti_dense",
        "seq2seq",
        "seq2seq_withouttargetmask",
        "contrasting",
        # baselines
        # "random_single",
        # "random_multi",
    ],
    "optimizer": ["adam"],
    "initializer": ["xavier_uniform_"],
    "learning_rate": ["2e-5", "3e-5", "5e-5"],
    "batch_size": [
        "16",
        # "32",
    ],  # ['16', '32'],
    "balancing": ["None"],  # ['None', 'lossweighting', 'oversampling'],
    "devmode": ["False"],
    "num_epoch": ["2", "3", "4"],
    "loss": [
        "crossentropy",
        "crossentropy_lsr",
        "sequence",
        "crossentropy_crossweight",
    ],
    # "spc_lm_representation_distilbert": ["mean_last"],
    # ['sum_last', 'sum_last_four', 'sum_last_two', 'sum_all', 'mean_last', 'mean_last_four', 'mean_last_two', 'mean_all'],
    # "spc_lm_representation": ["pooler_output"],
    # ['pooler_output', 'sum_last', 'sum_last_four', 'sum_last_two', 'sum_all', 'mean_last', 'mean_last_four', 'mean_last_two', 'mean_all'],
    # "spc_input_order": ["text_target"],  # 'target_text',
    # "aen_lm_representation": ["last"],
    # ['last', 'sum_last_four', 'sum_last_two', 'sum_all', 'mean_last_four'],  # 'mean_last_two', 'mean_all'],
    "eval_only_after_last_epoch": ["True"],
    "local_context_focus": ["cdm", "cdw"],
    "SRD": ["3", "4", "5"],
    "pretrained_model_name": ["default"],
    # ['default', 'bert_news_ccnc_10mio_3ep', 'laptops_and_restaurants_2mio_ep15', 'laptops_1mio_ep30', 'restaurants_10mio_ep3'],
    "state_dict": ["None"],
    # ['None', 'lcf_bert_acl14twitter_val_recall_avg_0.7349_epoch3', 'lcf_bert_semeval14laptops_val_recall_avg_0.7853_epoch3', 'lcf_bert_semeval14restaurants_val_recall_avg_0.7672_epoch2', 'lcf_bert_newstsc_val_recall_avg_0.5954_epoch3'],
    "single_targets": [
        "True"
    ],  # using conditions in controller.py, we have single_targets only for single target models
    "multi_targets": [
        "True"
    ],  # using conditions in controller.py, we have multi_targets only for multi target models
    "targetclasses": [
        "newsmtsc3",
        #"newsmtsc3strong",
        #"newsmtsc3weak",
    ],
    "knowledgesources": [
        "nrc_emotions", "mpqa_subjectivity", "bingliu_opinion", "liwc",
        "nrc_emotions mpqa_subjectivity", "nrc_emotions liwc",
        "nrc_emotions bingliu_opinion", "mpqa_subjectivity bingliu_opinion",
        "mpqa_subjectivity liwc", "bingliu_opinion liwc",
        "nrc_emotions mpqa_subjectivity bingliu_opinion",
        "nrc_emotions mpqa_subjectivity liwc",
        "nrc_emotions liwc bingliu_opinion",
        "liwc mpqa_subjectivity bingliu_opinion",
        "nrc_emotions mpqa_subjectivity bingliu_opinion liwc",
        "zeros",
    ],
    "is_use_natural_target_phrase_for_spc": [
        "True",
        "False"
    ],
    "default_lm": [
        BERT_BASE_UNCASED,
    ],
    "coref_mode_in_training": [
        "ignore",
        "in_targetmask",
        "additional_examples"
    ],
}
