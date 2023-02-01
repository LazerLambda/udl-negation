"""Script to Create Dataset."""

import checklist
import logging
import nltk
import re
import typing
import templates
import tqdm

from checklist.editor import Editor
from nltk.corpus import wordnet as wn
from sys import stdout

# Download NLTK Prerequisites
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_word_and_antonym(pos: str) -> list[tuple[str, str]]:
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

def fill_templates(wrd_ant_lst: list, template: str, mask_token: str, editor: Editor) -> list[tuple[str, str]]:
    """Fill Templates with Word and Antonym.
    
    Fills a list of words and antonyms into a provided template.

    :param wrd_ant_list: List of tuples including words (left) and their respective
        antonyms (right).
    :param template: Template that is to be filled with the words and antonyms. Must
        be a valid string for checklist's Editor class.
    :param mask_token: Special mask token of the used tokenizer of the main model.
    :paramm editor: Checklist Editor for formating strings.
    :returns: List of tuples consisting of (left) ground truth and (right) input.
    """
    return_list: list = []
    return_list.append(('y', 'x'))
    for (wrd, ant) in wrd_ant_lst:
        # TODO: Workaround. Fix later.
        # Account for a/an rule
        mask_token_ant: str = ant + mask_token
        mask_token_wrd: str = wrd + mask_token

        main = editor.template(template, word=[wrd], antonym=[ant]).data[0]

        masked0 = editor.template(template, word=[wrd], antonym=[mask_token_ant]).data[0]
        masked0: str = re.sub(mask_token_ant, mask_token, masked0)

        masked1 = editor.template(template, word=[mask_token_wrd], antonym=[ant]).data[0]
        masked1: str = re.sub(mask_token_wrd, mask_token, masked1)

        return_list.append((main, masked0))
        return_list.append((main, masked1))
    return return_list

def write_to_file(path: str, data: list) -> None:
    """Write Data to File.
    
    :param path: Path where the output file is written to.
    :param data: Data that is written to the file.
    """
    f = open(path, "w")
    f.writelines(map(lambda e: str(e[0]) + "|" + str(e[1]) + "\n", data))
    f.close()
    logging.info(str(len(data)) + f" lines written to file at {path}!")

def main():
    """Run Script."""
    editor: Editor = Editor()
    pos: list = ['NOUN']
    pos_wrd_ant_list: list = list(map(lambda e: (e, get_word_and_antonym(e)), pos))
    ret_list: list = list()
    for pos, wrd_ant_list in pos_wrd_ant_list:
        total: int = len(templates.TEMPLATES[pos])
        for i, temp in enumerate(templates.TEMPLATES[pos]):
            ret_list += fill_templates(wrd_ant_list, temp, '<mask>', editor)
            stdout.write(f"\r{pos}: {i+1}/{total}")
            stdout.flush()
    write_to_file('text.txt', ret_list)

if __name__ == "__main__":
    main()