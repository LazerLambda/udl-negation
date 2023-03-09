"""Script to Create Dataset."""

import logging
import os
import pathlib
import re
from sys import stdout

import nltk

try:
    from .templates import TEMPLATES
except Exception:
    from templates import TEMPLATES

from typing import List, Tuple

from checklist.editor import Editor
from nltk.corpus import wordnet as wn

# Download NLTK Prerequisites
nltk.download('wordnet')
nltk.download('omw-1.4')


def get_word_and_antonym(pos: str) -> List[Tuple[str, str]]:
    """Get List of Words and Antonyms.

    Get a list of all available part-of-speech words from wordnet and
    their respective antonym.

    :param pos: Part-of-speech tag. Necessary to request list from wordnet api.
    :returns: List of words and their respective antomys as tuples.
    """
    return_list: list = []
    for synset in list(wn.all_synsets(getattr(wn, pos))):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                word: str = str(lemma.name()).replace("_", " ")
                ant: str = str(lemma.antonyms()[0].name()).replace("_", " ")
                return_list.append((word, ant))
    logging.info("All words and their respective antonyms downloaded.")
    return return_list


def fill_templates(wrd_ant_lst: list, template: str, mask_token: str, editor: Editor, pos: str, amount: int = 1) -> List[Tuple[str, str]]:
    """Fill Templates with Word and Antonym.

    Fills a list of words and antonyms into a provided template.

    :param wrd_ant_lst: List of tuples including words (left) and their respective
        antonyms (right).
    :param template: Template that is to be filled with the words and antonyms. Must
        be a valid string for checklist's Editor class.
    :param mask_token: Special mask token of the used tokenizer of the main model.
    :param editor: Checklist Editor for formating strings.
    :param pos: Part-of-speech (e.g. 'ADJ', 'VERB', ...)
    :param amount: Amount of repeated samples.
    :returns: List of tuples consisting of (left) ground truth and (right) input.
    """
    return_list: list = []
    for (wrd, ant) in wrd_ant_lst:
        # TODO: Workaround. Fix later.
        # Account for a/an rule
        mask_token_ant: str = ant + mask_token
        mask_token_wrd: str = wrd + mask_token

        main = editor.template(template, word=[wrd], antonym=[ant]).data[0]

        masked0 = editor.template(template, word=[wrd], antonym=[mask_token_ant]).data[0]
        msk_tk_rgx: str = mask_token_ant.replace('[', '\[')
        msk_tk_rgx = msk_tk_rgx.replace(']', '\]')
        masked0_str: str = re.sub(msk_tk_rgx, mask_token, masked0)
        # if pos == "VERB":
        #     if bool(re.match("I", masked0_str)):
        #         masked0_str = re.sub(r" be ", " am ", masked0_str)
        #     elif bool(re.match("You|They|We", masked0_str)):
        #         masked0_str = re.sub(r" be ", " are ", masked0_str)
        #     else:
        #         masked0_str = re.sub(r" be ", " is ", masked0_str)

        masked1 = editor.template(template, word=[mask_token_wrd], antonym=[ant]).data[0]
        msk_tk_rgx: str = mask_token_wrd.replace('[', '\[')
        msk_tk_rgx = msk_tk_rgx.replace(']', '\]')
        masked1_str: str = re.sub(msk_tk_rgx, mask_token, masked1)
        for _ in range(amount):
            return_list.append((masked0_str, ant))
            return_list.append((masked1_str, wrd))
    return return_list


def write_to_file(path: str, data: list) -> None:
    """Write Data to File.

    :param path: Path where the output file is written to.
    :param data: Data that is written to the file.
    """
    head, _ = os.path.split(path)
    pathlib.Path(head).mkdir(parents=True, exist_ok=True)
    f = open(path, "w")
    f.writelines(map(lambda e: str(e[0]) + " [REF-BEG] " + str(e[1]) + " [REF-END]\n", data))
    f.close()
    logging.info(str(len(data)) + f" lines written to file at {path}!")


def main(data_path: str = '../data/processed/wn_neg_processed/data_wn_masked.txt', amount: int = 1) -> int:
    """Run Script.

    :param data_path: Path where dataset will be saved.
    :param amount: Amount of repeated samples.
    :returns: Length of dataset.
    """
    editor: Editor = Editor()
    pos: list = ['NOUN', 'ADJ']
    pos_wrd_ant_list: list = list(map(lambda e: (e, get_word_and_antonym(e)), pos))
    ret_list: list = list()
    counter: int = 0
    for pos, wrd_ant_list in pos_wrd_ant_list:
        total: int = len(TEMPLATES[pos])
        # counter += len(wrd_ant_list)
        for i, temp in enumerate(TEMPLATES[pos]):
            tmp: list = fill_templates(wrd_ant_list, temp, '[MASK]', editor, pos, amount)
            ret_list += tmp
            counter += len(tmp)
            stdout.write(f"\r{pos}: {i+1}/{total}")
            stdout.flush()
        stdout.flush()
    # ret_list.insert(0, ('y', 'x'))
    write_to_file(data_path, ret_list)
    return counter


if __name__ == "__main__":
    print("Amount: ", str(main()))
