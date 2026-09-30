"""Deterministic lexical tags extracted independently from each source text."""
import re
import unicodedata

# General language stop words, not benchmark entities or answer-derived rules.
STOP_WORDS = frozenset("""a an the and or but if then than so as at by for from in into of on to with without
is am are was were be been being do does did done have has had having will would shall should can could
may might must i me my mine myself we us our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves this that these those
there here who whom whose what which when where why how all any both each few more most other some such
no nor not only own same too very just now also about above after again against before below between
during further once until up down out off over under through while s t d ll m re ve don doesn didn isn
aren wasn weren wasn would wouldn couldn shouldn won ain let lets hey hi hello thanks thank please
really much many get got getting good great well like thing things something anything everything
nothing yes yeah okay ok one two ever never still even go going went say said tell told think know
want need able been make made see seen look looking new back lot much long way kind day today""".split())
WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[\u4e00-\u9fff]+")


def normalize_tags(values):
    result = set()
    for value in values:
        tag = re.sub(r"\s+", " ", unicodedata.normalize('NFKC', value).casefold()).strip(' .,:;')
        if tag and len(tag) <= 80:
            result.add(tag)
    return sorted(result)


class RuleTagger:
    identity = 'rule-keywords-v1:nfkc-stopwords-cjk-bigrams'

    def extract(self, texts, query=False):
        # Query and document use exactly the same rules; no shared batch context.
        output = []
        for text in texts:
            tags = set()
            normalized = unicodedata.normalize('NFKC', text).casefold().replace('’', "'")
            for word in WORDS.findall(normalized):
                if '\u4e00' <= word[0] <= '\u9fff':
                    tags.update(word[i:i+2] for i in range(len(word)-1))
                    if len(word) == 1:
                        tags.add(word)
                    continue
                word = word.removesuffix("'s")
                if word in STOP_WORDS or len(word) < 2 or word.isdecimal():
                    continue
                # Possessives and common English plural endings only. No invented synonyms.
                if len(word) > 4 and word.endswith('ies'):
                    word = word[:-3] + 'y'
                elif len(word) > 4 and word.endswith('s') and not word.endswith(('ss','us','is')):
                    word = word[:-1]
                if word not in STOP_WORDS:
                    tags.add(word)
            output.append(normalize_tags(tags))
        return output
