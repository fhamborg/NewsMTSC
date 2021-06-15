"""
Download a specific version of a finetuned model and place it in pretrained_models.
"""
import argparse


class Download:

    def __init__(self, own_model_name=None, version=None, force=False):
        pass

    @staticmethod
    def argument_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--own_model_name", default="grutsc", type=str)
        parser.add_argument(
            "--device",
            default=None,
            type=str,
            help="e.g., cuda:0; if None: any CUDA device will be used if available, else "
            "CPU",
        )
        parser.add_argument(
            "--version",
            default=None,
            type=str,
            help="version of the model to download, use --force to overwrite previous models"
        )
        parser.add_argument(
            "--force",
            default=False,
            type=bool,
            help="force the download of a model and overwrite potential previous versions"
        )
        return parser

    @classmethod
    def run_from_cmd(cls):
        args = vars(cls.argument_parser().parse_args())
        return cls(**args)

    @classmethod
    def run_defaults(cls):
        args = vars(cls.argument_parser().parse_args([]))
        return cls(**args)


if __name__ == "__main__":
    Download.run_from_cmd()
