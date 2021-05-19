import pandas as pd
import re
import spellcorrector
from string import punctuation
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer
import emoji


class Preprocessor:
    def __init__(self, **kwargs):
        self.make_lower = kwargs.get("make_lower", True)
        self.stopwords = kwargs.get("stopwords", None)
        self.stemmer = kwargs.get("stemmer", None)
        self.spell_corrector = kwargs.get("spell_corrector", None)

        self.treat_all_caps = kwargs.get("treat_all_caps", 'tag')
        self.treat_urls = kwargs.get("treat_urls", 'replace')
        self.treat_users = kwargs.get("treat_users", 'replace')
        self.treat_hashtags = kwargs.get("treat_hashtags", 'replace')
        self.treat_censored = kwargs.get("treat_censored", 'tag')
        self.treat_repeated = kwargs.get("treat_repeated", 'tag')
        self.treat_datetime = kwargs.get("treat_datetime", 'remove')
        self.treat_currencies = kwargs.get("treat_currencies", 'remove')

        # For debug
        self.verbose = kwargs.get("verbose", False)

        self.init_regexes()
        self.tags = ('url', 'hashtag', 'user', 'time', 'allcaps', 'repeat',
                     '\\url', '\\hashtag', '\\user', '\\time', '\\allcaps', '\\repeat')

    def init_regexes(self):
        self.all_caps_regex = r"(?<![#@$])\b([A-Z][A-Z ]{1,}[A-Z])\b"
        self.url_regex = r'(http://www\.|https://www\.|http://|https://)?[a-z0-9]+([\-.][a-z0-9]+)*' \
                         r'\.[a-z]{2,5}(:[0-9]{1,5})?(/\.*)?'
        self.user_regex = r"\B@\w\w+"
        self.hashtag_regex = r"(\B#)(\w\w+)"
        self.censored_regex = r"(\b\w+[\*|@]+\w+\b)"
        self.date_regex = r"(((\d{1,4}[\/\-.]){1,4}\d{1,4}) ?(AD|BC(E)?)?)|" \
                          r"((\d{1,4}(st|nd|rd|th)?)? ? (of)? ?" \
                          r"(January|February|March|April|May|June|July|August|September|October|Novmber|December)+" \
                          r" ?(the)? ?\d{,2}(st|nd|rd|th)? ?\d{0,4} ?(AD|BC(E)?)?)"
        self.time_regex = r"(\d{1,2} ?(AM|PM))|(\d{1,2}:\d{1,2}:?\d{1,2} ?(AM|PM)?)"
        self.date_time_regex = r"((<date>|<time>) *:?(<date>|<time>))|<time>|<date>"
        self.currency_regex = r"((?:[$\u20ac\u00a3\u00a2] ?\d+)(?:[\.,']\d+)*(?: ?[MmKkBb](?:n|(?:il(?:lion)?))?)?)|" \
                              r"((?:\d+(?:[\.,']\d+)*[$\u20ac\u00a3\u00a2])(?: ?[MmKkBb](?:n|(?:il(?:lion)?))?)?)"
        self.repeated = r"([!?.])([!?.]){1,}"
        self.emojis = re.compile(
                "(["
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F700-\U0001F77F"  # alchemical symbols
                "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                "\U00002702-\U000027B0"  # Dingbats
                "])"
                )

    def preprocess(self, text_data):
        """
        Preprocess all the text data

        :param text_data: A pandas data frame with all the text data
        :return: A pandas data frame with all the text data preprocessed
        """

        # Parse through each data element and preprocess it
        size = text_data.size
        for i in range(size):
            text_data.iat[i] = self.preprocess_line(text_data.iat[i])

        return text_data

    def preprocess_line(self, text):
        """
        Gets text and processes it depending on the arguments given in constructor

        :param text: The text to be processed
        :return processed_text: The text after processing
        """

        if len(text) == 0:
            return ''
        processed_text = text

        # Remove consecutive spaces
        processed_text = re.sub(r" {2,}", ' ', processed_text)

        # Unpack contractions
        processed_text = self.unpack_contractions(processed_text)

        # Process caps
        processed_text = self.process_occurrences(self.all_caps_regex, self.treat_all_caps, processed_text, '<allcaps>')

        # Make everything to lowercase
        if self.make_lower:
            processed_text = processed_text.lower()

        # Spell correction
        if self.spell_corrector is not None:
            processed_text = self.do_spell_correction(processed_text)

        # Remove stopwords from the text
        if self.stopwords is not None:
            processed_text = self.remove_stopwords(processed_text)

        # Stemming
        if self.stemmer is not None:
            processed_text = self.do_stemming(processed_text)

        processed_text = self.treat_emoji(processed_text)

        # Process urls
        processed_text = self.process_occurrences(self.url_regex, self.treat_urls, processed_text, '<url>')

        # Process user mentions
        processed_text = self.process_occurrences(self.user_regex, self.treat_users, processed_text, '<user>')

        # Process hashtags
        processed_text = self.process_hashtags(self.hashtag_regex, self.treat_hashtags, processed_text, '<hashtag>')

        # Process censored words
        processed_text = self.process_occurrences(self.censored_regex, self.treat_censored, processed_text,
                                                  '<censored>')

        # Process repeated occurrences
        processed_text = self.process_repeated(self.repeated, self.treat_repeated, processed_text, '<repeat>')

        # Process dates and times
        processed_text = self.process_datetime(processed_text)

        # Process currencies
        processed_text = self.process_occurrences(self.currency_regex, self.treat_currencies, processed_text,
                                                  '<number>')

        if self.verbose:
            print('{} ==> {}'.format(text, processed_text))

        return processed_text

    def treat_emoji(self, processed_text):
        """
        Replace empjis with the <emoji> tag. Uses the emoji package to get an emoji regex 
        """
        processed_text = emoji.get_emoji_regexp().sub('<emoji>', processed_text)

        return processed_text

    def process_occurrences(self, regex, opt, processed_text, tag):
        if opt == 'remove':
            processed_text = re.sub(regex, ' ', processed_text)
        elif opt == 'replace':
            processed_text = re.sub(regex, r" " + tag + r" ", processed_text)
        elif opt == 'tag':
            processed_text = re.sub(regex, r" " + tag + r" \g<0> " + tag + r" ", processed_text)

        return processed_text

    def process_hashtags(self, regex, opt, processed_text, tag):
        if opt == 'remove':
            processed_text = re.sub(regex, ' ', processed_text)
        elif opt == 'replace':
            processed_text = re.sub(regex, tag + r" ", processed_text)
        elif opt == 'tag':
            processed_text = re.sub(regex, r" " + tag + r" \g<2> " + tag + r" ", processed_text)

        return processed_text

    def process_repeated(self, regex, opt, processed_text, tag):
        if opt == 'remove':
            return processed_text
        else:
            if opt == 'replace':
                processed_text = re.sub(regex, tag, processed_text)
            elif opt == 'tag':
                processed_text = re.sub(regex, r" " + tag + r" \1\2 " + tag + r" ", processed_text)

        return processed_text

    def process_datetime(self, processed_text):
        # Remove date and times or change them with <datetime> tag
        if self.treat_datetime == 'replace':
            processed_text = re.sub(self.date_regex, ' <date> ', processed_text)
            processed_text = re.sub(self.time_regex, ' <time> ', processed_text)
            processed_text = re.sub(r" +|\t", " ", processed_text)
            processed_text = re.sub(self.date_time_regex, ' <time> ', processed_text)
            processed_text = re.sub(r"(<time>)", " <time> ", processed_text)
        elif self.treat_datetime == 'remove':
            processed_text = re.sub(self.date_regex, ' <date> ', processed_text)
            processed_text = re.sub(self.time_regex, ' <time> ', processed_text)
            processed_text = re.sub(r" +|\t", " ", processed_text)
            processed_text = re.sub(self.date_time_regex, ' ', processed_text)

        return processed_text

    def remove_stopwords(self, processed_text):
        if not processed_text:
            return ''

        # Split to words
        tokens = processed_text.split()
        filtered = []
        i = 0
        tokens_count = len(tokens)
        found = False
        while i < tokens_count:
            # Do not split our inserted tags
            if tokens[i] == '<' and i == tokens_count - 1:
                filtered.append(' ' + tokens[i])
            elif tokens[i] == '<' and tokens[i + 1] in self.tags:
                if not found:
                    filtered.append(' ' + tokens[i])
                else:
                    filtered.append(tokens[i])
                filtered.append(tokens[i + 1])
                filtered.append(tokens[i + 2])
                i = i + 3
                found = not found
                if i < tokens_count:
                    filtered.append(tokens[i])
            else:
                if tokens[i] not in self.stopwords:
                    if i == 0:
                        filtered.append(tokens[i])
                    else:
                        # Avoid splitting punctuation (at least unless we decide to do something else)
                        if filtered:
                            if filtered[-1] in punctuation and tokens[i] in punctuation:
                                filtered.append(tokens[i])
                            else:
                                filtered.append(' ' + tokens[i])
                        else:
                            filtered.append(tokens[i])
            i = i + 1

        if not filtered:  # list is empty
            processed_text = ''
        else:
            processed_text = filtered[0]
            for word in filtered[1:]:
                processed_text = processed_text + word

        return processed_text

    def do_stemming(self, processed_text):
        # Split to words
        tokens = processed_text.split()
        stemmed = []
        i = 0
        tokens_count = len(tokens)
        found = False
        while i < tokens_count:
            # Do not split our inserted tags
            if tokens[i] == '<' and tokens[i + 1] in self.tags:
                if not found:
                    stemmed.append(' ' + tokens[i])
                else:
                    stemmed.append(tokens[i])
                stemmed.append(tokens[i + 1])
                stemmed.append(tokens[i + 2])
                i = i + 3
                found = not found
                if i < tokens_count:
                    stemmed.append(tokens[i])
                i = i + 1
            else:
                if i == 0:
                    stemmed.append(self.stemmer.stem(tokens[i]))
                else:
                    # Avoid splitting punctuation (at least unless we decide to do something else
                    if stemmed[-1] in punctuation and tokens[i] in punctuation:
                        stemmed.append(self.stemmer.stem(tokens[i]))
                    else:
                        stemmed.append(' ' + self.stemmer.stem(tokens[i]))
                i = i + 1

        if not stemmed:  # list is empty
            processed_text = ''
        else:
            processed_text = stemmed[0]
            for word in stemmed[1:]:
                processed_text = processed_text + word

        return processed_text

    def do_spell_correction(self, processed_text):
        tokens = processed_text.split()
        corrected = []
        i = 0
        tokens_count = len(tokens)
        found = False
        while i < tokens_count:
            # Do not split our inserted tags
            if tokens[i] == '<' and tokens[i + 1] in self.tags:
                if not found:
                    corrected.append(' ' + tokens[i])
                else:
                    corrected.append(tokens[i])
                corrected.append(tokens[i + 1])
                corrected.append(tokens[i + 2])
                i = i + 3
                found = not found
                if i < tokens_count:
                    corrected.append(tokens[i])
                i = i + 1
            else:
                if i == 0:
                    corrected.append(self.spell_corrector.correction(tokens[i]))
                else:
                    # Avoid splitting punctuation (at least unless we decide to do something else
                    if corrected[-1] in punctuation and tokens[i] in punctuation:
                        corrected.append(self.spell_corrector.correction(tokens[i]))
                    else:
                        corrected.append(' ' + self.spell_corrector.correction(tokens[i]))
                i = i + 1

        if not corrected:  # list is empty
            processed_text = ''
        else:
            processed_text = corrected[0]
            for word in corrected[1:]:
                processed_text = processed_text + word

        return processed_text

    @staticmethod
    def unpack_contractions(processed_text):
        """Important Note: The function is taken from textacy (https://github.com/chartbeat-labs/textacy).
        See textacy.preprocess.unpack_contractions(text) -> http://textacy.readthedocs.io/en/ atest/api_reference.html
        The reason that textacy is not added as a dependency is to avoid having the user to install it 's dependencies
        (such as SpaCy), in order to just use this function.
        """
        # standard
        processed_text = re.sub(
            r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|"
            r"[Ww]ere|[Ww]ould)n't", r"\1\2 not", processed_text)
        processed_text = re.sub(
            r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll",
            r"\1\2 will", processed_text)
        processed_text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are",
                                processed_text)
        processed_text = re.sub(
            r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve",
            r"\1\2 have", processed_text)
        # non-standard
        processed_text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", processed_text)
        processed_text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", processed_text)
        processed_text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", processed_text)
        processed_text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", processed_text)
        processed_text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", processed_text)
        processed_text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", processed_text)
        return processed_text