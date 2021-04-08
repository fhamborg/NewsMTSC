BERT_BASE_UNCASED = "bert-base-uncased"
ROBERTA_BASE = "roberta-base"
XLNET_BASE_CASED = "xlnet-base-cased"
ALBERT_BASE = "albert-base-v2"
ALBERT_LARGE = "albert-large-v2"
ALBERT_XLARGE = "albert-xlarge-v2"
ALBERT_XXLARGE = "albert-xxlarge-v2"
__DEFAULT_LM = None


def set_default_lm(new_name: str):
    global __DEFAULT_LM
    __DEFAULT_LM = new_name


def get_default_lm():
    return __DEFAULT_LM


FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS = "text_ids_with_special_tokens"
FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK = (
    FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS + "_target_mask"
)
FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_SELECTED_KNOWLEDGE_SOURCES = (
    FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS + "_selectedknowledgesources"
)
FIELD_IS_OVERFLOW = "is_overflow"
FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS = (
    "text_then_target_ids_with_special_tokens"
)
# we used to have text-then-target target mask here, but won't use it,
# since it would be identical to the text target mask (since we only
# want to mark the target within the text, but not in the 2nd target
# component)
# FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK = (
#    FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS + "_target_mask"
# )
# same for knowledge sources
# FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_KNOWLEDGE_SOURCES = (
#         FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS + "_selectedknowledgesources"
# )
FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_SEGMENT_IDS = (
    FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS + "_segment_ids"
)
FIELD_TARGET_IDS_WITH_SPECIAL_TOKENS = "target_ids_with_special_tokens"
FIELD_SYNTAX_HOP_DISTANCE_TO_TARGET = "syntax_hop_distance_to_target"
FIELD_SYNTAX_DEPENDENCY_MATRIX = "syntax_dependency_matrix"
