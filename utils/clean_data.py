import os
import nltk
import re
from random import shuffle


def replace_email_addresses(text):
    replaced_email = []
    for item in text:
        emails = re.findall('\S+@\S+', item)
        replace_text = item
        for email in emails:
            replace_text = replace_text.replace(email, ' EMAIL ')
        replaced_email.append(replace_text)
    return replaced_email


def replace_url(text):
    replaced_url = []
    regex = r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"
    for item in text:
        urls = re.findall(regex, item)
        replace_text = item
        for url in urls:
            replace_text = replace_text.replace(url, ' URL ')
        replaced_url.append(replace_text)
    return replaced_url


def replace_link(text):
    replaced_text = []
    for item in text:
        replacement = item
        replacement = replacement.replace('<<link>>', ' LINK ')
        replacement = replacement.replace('<<LINK>>', ' LINK ')
        replacement = replacement.replace('<link>', ' LINK ')
        replacement = replacement.replace('<<LINK>>', ' LINK ')
        replaced_text.append(replacement)
    return replaced_text


def replace_numbers(text):
    num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
    replaced_text = []
    for item in text:
        new_text = []
        words = item.split(' ')
        for word in words:
            if num_regex.match(word):
                new_text.append('NUM')
            else:
                new_text.append(word)
        replaced_text.append(' '.join(new_text))
    return replaced_text


def remove_html(text):
    clean = []
    for item in text:
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', item)
        clean.append(cleantext)
    return clean


def remove_css(text):
    print("TOTAL EMAILS: {}".format(len(text)))
    clean = []
    regex = re.compile(r'\b(\w+):(\w+)(?=;)')
    for item in text:
        iter = re.finditer(regex, item)
        indices = [(m.start(0), m.end(0)) for m in iter]
        if indices:
            continue
        else:
            clean.append(item)
    print("CSS REMOVED EMAILS: {}".format(len(clean)))
    return clean


def remove_equals(text):
    replaced = []
    for item in text:
        item = item.replace('= ', '')
        replaced.append(item)
    return replaced


def remove_nonsense(text):
    replaced = []
    for item in text:
        words = item.split(' ')
        long_strings = []
        for word in words:
            if len(word) > 30:
                long_strings.append(word)
        for string_ in long_strings:
            item = item.replace(string_, ' LONG ')
        replaced.append(item)
    return replaced


def remove_excess_spaces(text):
    replaced = []
    for item in text:
        item = ' '.join(item.split())
        replaced.append(item)
    return replaced


def remove_new_line(text):
    removed_newline = []
    for item in text:
        item = item.replace('\n', ' ')
        item = item.replace('\xa0', ' ')
        removed_newline.append(item)
    return removed_newline


def get_text(root_path):
    emails = []
    for email_text in os.listdir(root_path):
        email_file = root_path + email_text
        email = open(email_file, 'r')
        text = email.read()
        emails.append(str(text))
    return emails


def tokenize_text(text):
    tokenized = []
    for item in text:
        tokens = nltk.word_tokenize(item)
        if len(tokens) > 300:
            tokenized.append(tokens[:300])
        else:
            tokenized.append(tokens)
    return tokenized


def append_label(text, phish=False):
    label_appended = []
    if not phish:
        for item in text:
            label_appended.append((item, 0))
    else:
        for item in text:
            label_appended.append((item, 1))
    return label_appended


def combine_phish_and_legit(legit, phish):
    legit.extend(phish)
    shuffle(legit)
    return legit


def process_legit_phish_data(legit_path='ISWPA2.0 Train Data/IWSPA2.0_Training_No_Header/legit/', phish_path='ISWPA2.0 Train Data/IWSPA2.0_Training_No_Header/phish/'):
    legit_text = get_text(legit_path)
    phish_text = get_text(phish_path)
    legit_text = remove_new_line(legit_text)
    phish_text = remove_new_line(phish_text)
    legit_text = replace_email_addresses(legit_text)
    phish_text = replace_email_addresses(phish_text)
    legit_text = replace_url(legit_text)
    phish_text = replace_url(phish_text)
    legit_text = replace_link(legit_text)
    phish_text = replace_link(phish_text)
    legit_text = remove_html(legit_text)
    phish_text = remove_html(phish_text)
    legit_text = remove_nonsense(legit_text)
    phish_text = remove_nonsense(phish_text)
    legit_text = remove_excess_spaces(legit_text)
    phish_text = remove_excess_spaces(phish_text)
    legit_text = remove_css(legit_text)
    phish_text = remove_css(phish_text)
    legit_text = replace_numbers(legit_text)
    phish_text = replace_numbers(phish_text)
    legit_text = remove_equals(legit_text)
    phish_text = remove_equals(phish_text)
    legit_text = tokenize_text(legit_text)
    phish_text = tokenize_text(phish_text)
    legit_text = append_label(legit_text, phish=False)
    phish_text = append_label(phish_text, phish=True)
    combined = combine_phish_and_legit(legit_text, phish_text)
    return combined


if __name__ == '__main__':
    process_legit_phish_data()
