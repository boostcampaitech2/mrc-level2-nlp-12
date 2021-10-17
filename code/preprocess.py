import pandas as pd
import os
import re
from soynlp.normalizer import *
from pykospacing import Spacing

class Preprocess():

    PERMIT_REMOVE_LANGS = [
        'arabic',
        'russian',
        # 'english'
    ]

    def __init__(self, sents, langs: list):
        self.sents = sents
        self.spacing = Spacing()
        self.langs = langs  # 제거 대상 언어

        self.ord_list = []
        for lang in langs:
            lang = lang.lower()
            if lang not in Preprocess.PERMIT_REMOVE_LANGS:
                raise ValueError('[ Not Removable Lang ] 제거 대상이 아닌 언어입니다.')

            if lang == 'arabic':
                self.ord_list.append(('0600', '06FF'))
            elif lang == 'russian':
                self.ord_list.append(('0400', '04FF'))
            # elif lang == 'english':
            #     self.ord_list.append(('0041', '007A'))

        print('--- Removable Langs ---')
        print(self.ord_list)

    def proc_preprocessing(self):
        """
        A function for doing preprocess
        """
        self.remove_html()
        self.remove_email()
        self.remove_hashtag()
        self.remove_user_mention()
        self.remove_url()
        self.remove_bad_char()
        self.remove_press()
        self.remove_copyright()
        self.remove_photo_info()
        self.remove_useless_breacket()
        self.remove_repeat_char()
        self.clean_punc()
        self.remove_linesign()
        self.remove_language()
        self.remove_repeated_spacing()
        # self.spacing_sent() # spacing

    def remove_html(self):
        """
        A function for removing html tags
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = re.sub(r"<[^>]+>\s+(?=<)|<[^>]+>", "", sent).strip()
            if sent:
                # print(sent)
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_email(self):
        """
        A function for removing email address
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_hashtag(self):
        """
        A function for removing hashtag
        """
        preprocessed_sents = []
        for sent in self.sents:
            # \S => [^ \t\n\r\f\v]
            sent = re.sub(r"#\S+", "", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_user_mention(self):
        """
        A function for removing mention tag
        """
        preprocessed_sents = []
        for sent in self.sents:
            # \w => [a-zA-Z0-9_] (문자 + 숫자)
            # \W => [^a-zA-Z0-9_] (문자 + 숫자 외)
            sent = re.sub(r"@\w+", "", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_url(self):
        """
        A function for removing URL address
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", "", sent).strip()
            sent = re.sub(r"pic\.(\w+\.)+\S*", "", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_bad_char(self):
        """
        A function for removing raw unicode including unk
        """
        bad_chars = {"\u200b": "", "…": " ... ", "\ufeff": ""}
        preprcessed_sents = []
        for sent in self.sents:
            for bad_char in bad_chars:
                sent = sent.replace(bad_char, bad_chars[bad_char])
            sent = re.sub(r"[\+á?\xc3\xa1]", "", sent)
            if sent:
                preprcessed_sents.append(sent)
        return preprcessed_sents

    def remove_press(self):
        """
        A function for removing press information
        """
        re_patterns = [
            r"\([^(]*?(뉴스|경제|일보|미디어|데일리|한겨례|타임즈|위키트리)\)",
            r"[가-힣]{0,4} (기자|선임기자|수습기자|특파원|객원기자|논설고문|통신원|연구소장) ",
            r"[가-힣]{1,}(뉴스|경제|일보|미디어|데일리|한겨례|타임|위키트리)",
            r"\(\s+\)",
            r"\(=\s+\)",
            r"\(\s+=\)",
        ]

        preprocessed_sents = []
        for sent in self.sents:
            for re_pattern in re_patterns:
                sent = re.sub(re_pattern, "", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_copyright(self):
        """
        A function for removing signs of copyrights
        """
        re_patterns = [
            r"\<저작권자(\(c\)|ⓒ|©|\(Copyright\)|(\(c\))|(\(C\))).+?\>",
            r"저작권자\(c\)|ⓒ|©|(Copyright)|(\(c\))|(\(C\))"
        ]
        preprocessed_sents = []
        for sent in self.sents:
            for re_pattern in re_patterns:
                sent = re.sub(re_pattern, "", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_photo_info(self):
        """
        A function for removing image captions
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = re.sub(r"\(출처 ?= ?.+\) |\(사진 ?= ?.+\) |\(자료 ?= ?.+\)| \(자료사진\) |사진=.+기자 ", "", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_useless_breacket(self):
        """
        A function for removing meaningless words in wikipedia
        수학(數學,) => 수학(數學)
        """
        bracket_pattern = re.compile(r"\((.*?)\)")
        preprocessed_text = []
        for text in self.sents:
            modi_text = ""
            text = text.replace("()", "")
            brackets = bracket_pattern.search(text)
            if not brackets:
                if text:
                    preprocessed_text.append(text)
                    continue
            replace_brackets = {}
            while brackets:
                index_key = str(brackets.start()) + "," + str(brackets.end())
                bracket = text[brackets.start() + 1: brackets.end() - 1]
                infos = bracket.split(",")
                modi_infos = []
                for info in infos:
                    info = info.strip()
                    if len(info) > 0:
                        modi_infos.append(info)
                if len(modi_infos) > 0:
                    replace_brackets[index_key] = "(" + ", ".join(modi_infos) + ")"
                else:
                    replace_brackets[index_key] = ""
                brackets = bracket_pattern.search(text, brackets.start() + 1)
            end_index = 0
            for index_key in replace_brackets.keys():
                start_index = int(index_key.split(",")[0])
                modi_text += text[end_index:start_index]
                modi_text += replace_brackets[index_key]
                end_index = int(index_key.split(",")[1])
            modi_text += text[end_index:]
            modi_text = modi_text.strip()
            if modi_text:
                preprocessed_text.append(modi_text)
        self.sents = preprocessed_text

    def remove_repeat_char(self):
        """
        A function for removing repeated char over > 3
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = repeat_normalize(sent, num_repeats=3).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def clean_punc(self):
        """
        A function for removing useless punctuation
        """
        punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                         "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                         '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-',
                         'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

        preprocessed_sents = []
        for sent in self.sents:
            for p in punct_mapping:
                sent = sent.replace(p, punct_mapping[p])
            sent = sent.strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_repeated_spacing(self):
        """
        A function for reducing whitespaces into one
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = re.sub(r"\s+", " ", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_linesign(self):
        """
        A function for removing line sings like \n
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = re.sub(r"[\n\t\r\v\f\\\\n\\t\\r\\v\\f/[{2,}]{2,}]", "", sent)
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def spacing_sent(self):
        """
        A function for spacing properly
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = self.spacing(sent)
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_language(self):
        """
        A function for removing other langs
        """
        preprocessed_sents = []
        for sent in self.sents:
            return_sentence = ''
            for ord_pair in self.ord_list:
                a = int(ord_pair[0], 16)
                b = int(ord_pair[1], 16)
                for i, w in enumerate(sent):
                    if a <= ord(w) and ord(w) <= b:
                        continue
                    return_sentence += w
            preprocessed_sents.append(return_sentence)
        self.sents = preprocessed_sents