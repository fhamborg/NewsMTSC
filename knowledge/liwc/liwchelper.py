from .dic import read_dic
from .trie import build_trie, search_trie

try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("liwc").version
except Exception:
    __version__ = None


def load_token_parser(filepath="knowledge/liwc/data/LIWC2015_English.dic"):
    """
    Reads a LIWC lexicon from a file in the .dic format, returning a tuple of
    (parse, category_names), where:
    * `parse` is a function from a token to a list of strings (potentially
      empty) of matching categories
    * `category_names` is a list of strings representing all LIWC categories in
      the lexicon
    """
    lexicon, category_names = read_dic(filepath)
    trie = build_trie(lexicon)

    def parse_token(token):
        return search_trie(trie, token)

    return parse_token, category_names
