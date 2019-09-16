import glob
import re
from itertools import chain
from multiprocessing import Pool

import fire as fire
from tqdm import tqdm


def splitkeepsep(s, sep):
    cleaned = []
    s = re.split("(%s)" % re.escape(sep), s)
    for _ in s:
        if _ != '' and _ != sep:
            cleaned.append(sep + _)
    return cleaned


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_special_chars(text, char_list):
    for char in char_list:
        text = text.replace(char, '')
    return text.replace(u'\xa0', u' ')


def process_wiki_files(wiki_file):
    chars = ['\n']
    global sw

    with open(wiki_file, encoding='utf-8') as f:
        content = f.read()

    articles = splitkeepsep(content, '<doc id=')
    res = []

    for article in articles:
        article = remove_special_chars(remove_html_tags(article), chars)
        res.append(article)

    return res


def list_multiprocessing(param_lst, func, **kwargs):
    workers = kwargs.pop('workers')

    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i, params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))

    # lists do not need such sorting, but this can be useful later
    result = sorted(result, key=lambda x: x[0])
    return [_[1] for _ in result]


def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params, **kwargs)


def main(wiki_dir, output_path):
    wiki_files = []
    for filename in glob.iglob(f'{wiki_dir}/*/*', recursive=True):
        wiki_files.append(filename)

    articles = list_multiprocessing(wiki_files, process_wiki_files, workers=4)

    with open(output_path, 'w', encoding='utf-8') as f:
        for article in tqdm(chain.from_iterable(articles)):
            f.write(article + '\n\n')


if __name__ == '__main__':
    sw = None
    fire.Fire(main)
