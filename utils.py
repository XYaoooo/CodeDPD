import re

def remove_extra_spaces(str):
    clean = lambda x: re.sub(r'[ \t]{2,}', ' ', x)
    cleaned_str = clean(str)
    return cleaned_str
