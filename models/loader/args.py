import argparse
import os
from configs.kb_configs import *


# Additional argparse types
def path(string):
    if not string:
        return ''
    s = os.path.expanduser(string)
    if not os.path.exists(s):
        raise argparse.ArgumentTypeError(f'No such file or directory: "{string}"')
    return s


def file_path(string):
    if not string:
        return ''
    s = os.path.expanduser(string)
    if not os.path.isfile(s):
        raise argparse.ArgumentTypeError(f'No such file: "{string}"')
    return s


def dir_path(string):
    if not string:
        return ''
    s = os.path.expanduser(string)
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError(f'No such directory: "{string}"')
    return s


parser = argparse.ArgumentParser(prog='langchain-ChatGLM',
                                 description='About langchain-ChatGLM, local knowledge based ChatGLM with langchain ｜ '
                                             '基于本地知识库的 ChatGLM 问答')

parser.add_argument('--no-remote-model', action='store_true', help='remote in the model on '
                                                                   'loader checkpoint, '
                                                                   'if your load local '
                                                                   'model to add the ` '
                                                                   '--no-remote-model`')


args = parser.parse_args([])
# Generares dict with a default value for each argument
DEFAULT_ARGS = vars(args)
