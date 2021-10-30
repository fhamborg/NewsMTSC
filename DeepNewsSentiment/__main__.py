import argparse

from DeepNewsSentiment.download import Download

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='DeepNewsAnalysis')
    subparsers = parser.add_subparsers(dest='action')

    subparser_download = subparsers.add_parser('download', help=Download.add_subparser.__doc__)
    Download.add_subparser(subparser_download)

    args = parser.parse_args()
    action = args.action
    del args.action

    if action == 'download':
        Download.run_from_parser(args)
