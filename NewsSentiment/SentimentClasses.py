from typing import Iterable


class SentimentClasses:
    FILLUP_POLARITY_VALUE = -100
    FILLUP_POLARITY_LABEL = "fillup"
    SENTIMENT_CLASSES = None

    @staticmethod
    def initialize(sentiment_classes: dict):
        SentimentClasses.SENTIMENT_CLASSES = sentiment_classes

    @staticmethod
    def get_num_classes():
        return len(SentimentClasses.SENTIMENT_CLASSES)

    @staticmethod
    def __evaluate_boundary(given_value: float, boundary: tuple):
        operator = boundary[0]
        value = boundary[1]
        if operator == "<=":
            return given_value <= value
        elif operator == "<":
            return given_value < value
        elif operator == ">=":
            return given_value >= value
        elif operator == ">":
            return given_value > value
        elif operator == "==":
            return given_value == value
        else:
            raise ValueError

    @staticmethod
    def __evaluate_boundaries_of_class(
        given_value: float, sentiment_boundaries: Iterable[tuple]
    ):
        assert len(sentiment_boundaries) >= 1
        for boundary in sentiment_boundaries:
            is_valid = SentimentClasses.__evaluate_boundary(given_value, boundary)
            if not is_valid:
                return False
        return True

    @staticmethod
    def __get_legacy_information():
        # self.polarity_associations = {"positive": 2, "neutral": 1, "negative": 0}
        # self.polarity_associations_inv = {2: "positive", 1: "neutral", 0: "negative"}
        # self.sorted_expected_label_values = [0, 1, 2]
        # self.sorted_expected_label_names = ["negative", "neutral", "positive"]

        sentiment_labels = list(SentimentClasses.SENTIMENT_CLASSES.keys())
        sentiment_normalized_values = []
        for label in sentiment_labels:
            sentiment_normalized_values.append(
                SentimentClasses.SENTIMENT_CLASSES[label]["normalized_polarity"]
            )

        polarity_associations = {}
        polarity_associations_inv = {}
        for label, value in zip(sentiment_labels, sentiment_normalized_values):
            polarity_associations[label] = value
            polarity_associations_inv[value] = label

        return {
            "polarity_associations": polarity_associations,
            "polarity_associations_inv": polarity_associations_inv,
            "sorted_expected_label_values": sentiment_normalized_values,
            "sorted_expected_label_names": sentiment_labels,
        }

    @staticmethod
    def get_sorted_expected_label_names():
        return SentimentClasses.__get_legacy_information()[
            "sorted_expected_label_names"
        ]

    @staticmethod
    def get_sorted_expected_label_values():
        return SentimentClasses.__get_legacy_information()[
            "sorted_expected_label_values"
        ]

    @staticmethod
    def get_polarity_associations():
        return SentimentClasses.__get_legacy_information()["polarity_associations"]

    @staticmethod
    def get_polarity_associations_inverse():
        return SentimentClasses.__get_legacy_information()["polarity_associations_inv"]

    @staticmethod
    def __find_sentiment_class(polarity: float):
        resulting_class = None
        for sentiment_label, info in SentimentClasses.SENTIMENT_CLASSES.items():
            sentiment_boundaries = info["boundaries"]
            sentiment_normalized_polarity = info["normalized_polarity"]
            is_in_class_boundaries = SentimentClasses.__evaluate_boundaries_of_class(
                polarity, sentiment_boundaries
            )
            if is_in_class_boundaries:
                # check polarity is not in another class, too
                assert (
                    resulting_class is None
                ), f"overlapping sentiment classes; previous class: {resulting_class}"
                resulting_class = (sentiment_label, sentiment_normalized_polarity)

        # check that a class was found
        assert resulting_class, f"result is not defined for polarity: {polarity}"

        return resulting_class

    @staticmethod
    def polarity2label(polarity: float) -> str:
        if polarity == SentimentClasses.FILLUP_POLARITY_VALUE:
            return SentimentClasses.FILLUP_POLARITY_LABEL

        sentiment_class = SentimentClasses.__find_sentiment_class(polarity)
        label = sentiment_class[0]
        return label

    @staticmethod
    def polarity2normalized_polarity(polarity: float) -> int:
        if polarity == SentimentClasses.FILLUP_POLARITY_VALUE:
            return int(SentimentClasses.FILLUP_POLARITY_VALUE)

        sentiment_class = SentimentClasses.__find_sentiment_class(polarity)
        normalized_polarity = sentiment_class[1]
        return normalized_polarity

    @staticmethod
    def Sentiment3ForNewsMtsc():
        sentiment_classes = {
            "positive": {
                "boundaries": [(">=", 5), ("<=", 7)],
                "normalized_polarity": 2,
            },
            "neutral": {"boundaries": [(">", 3), ("<", 5)], "normalized_polarity": 1},
            "negative": {
                "boundaries": [(">=", 1), ("<=", 3)],
                "normalized_polarity": 0,
            },
        }
        SentimentClasses.initialize(sentiment_classes)

    @staticmethod
    def SentimentStrong3ForNewsMtsc():
        sentiment_classes = {
            "positive": {
                "boundaries": [(">=", 6), ("<=", 7)],
                "normalized_polarity": 2,
            },
            "neutral": {"boundaries": [(">", 2), ("<", 6)], "normalized_polarity": 1},
            "negative": {
                "boundaries": [(">=", 1), ("<=", 2)],
                "normalized_polarity": 0,
            },
        }
        SentimentClasses.initialize(sentiment_classes)

    @staticmethod
    def SentimentWeak3ForNewsMtsc():
        sentiment_classes = {
            "positive": {
                "boundaries": [(">=", 4.5), ("<=", 7)],
                "normalized_polarity": 2,
            },
            "neutral": {
                "boundaries": [(">", 3.5), ("<", 4.5)],
                "normalized_polarity": 1,
            },
            "negative": {
                "boundaries": [(">=", 1), ("<=", 3.5)],
                "normalized_polarity": 0,
            },
        }
        SentimentClasses.initialize(sentiment_classes)
