"""Bounded deterministic query-time interpretation. Never uses wall-clock time."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import calendar
import re
from .temporal import TemporalNormalizer

WORDS = dict(zip('one two three four five six seven eight nine ten eleven twelve'.split(),range(1,13)))
NUM = r'(?:\d{1,3}|'+'|'.join(WORDS)+')'
WEEKDAYS = 'monday tuesday wednesday thursday friday saturday sunday'.split()
PATTERN = re.compile(
    r'\blast (?:'+'|'.join(WEEKDAYS)+r')\b|上周[一二三四五六日天]|'
    r'\b(?:'+NUM+r') (?:days?|weeks?) ago\b|'
    r'\b(?:past|last) '+NUM+r' months?\b|'
    r'\b(?:yesterday|today|tomorrow|last week|last month|last year)\b|昨天|今天|明天|上个月|'
    r'\b\d{4}-\d{1,2}-\d{1,2}\b|\d{4}年\d{1,2}月\d{1,2}日',re.I)


def validate_zone(value):
    try: ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc: raise ValueError('Invalid reference timezone') from exc
    return value


def month_shift(day, delta):
    serial=day.year*12+day.month-1+delta
    year,month=serial//12,serial%12+1
    return date(year,month,min(day.day,calendar.monthrange(year,month)[1]))


@dataclass(frozen=True)
class QueryTime:
    expression: str
    start: date
    end: date

    def description(self):
        def display(d): return f'{d.isoformat()} ({WEEKDAYS[d.weekday()].capitalize()})'
        return display(self.start) if self.start==self.end else display(self.start)+' through '+display(self.end)


def resolve_query_time(query, reference_time=None, reference_timezone='UTC'):
    matches=list(PATTERN.finditer(query))
    # Multiple constraints, comparisons and negations require an explicit
    # operator model; do not mistake them for a positive point-time target.
    if len(matches)!=1 or re.search(r'\b(before|after|between|until|since|not|except)\b|之前|之后|不是|除外',query,re.I): return None
    text=matches[0].group(); lower=text.lower()
    try:
        norm=TemporalNormalizer.normalize(lower)
        if norm['start']:
            return QueryTime(text,date.fromisoformat(norm['start']),date.fromisoformat(norm['end']))
        if reference_time is None: return None
        anchor=datetime.fromtimestamp(reference_time/1000,ZoneInfo(reference_timezone)).date()
        if lower.startswith('last ') and lower[5:] in WEEKDAYS:
            days=(anchor.weekday()-WEEKDAYS.index(lower[5:]))%7 or 7
            start=end=anchor-timedelta(days=days)
        elif text.startswith('上周'):
            index='一二三四五六日'.find(text[-1].replace('天','日'))
            start=end=anchor-timedelta(days=anchor.weekday()+7-index)
        elif lower=='last week':
            start=anchor-timedelta(days=anchor.weekday()+7);end=start+timedelta(days=6)
        elif m:=re.fullmatch('('+NUM+r') (days?|weeks?) ago',lower):
            n=int(m[1]) if m[1].isdigit() else WORDS[m[1]]
            start=end=anchor-timedelta(days=n*(7 if m[2].startswith('week') else 1))
        elif m:=re.fullmatch(r'(?:past|last) ('+NUM+r') months?',lower):
            n=int(m[1]) if m[1].isdigit() else WORDS[m[1]]
            start,end=month_shift(anchor,-n),anchor
        else:
            # The normalizer consumes UTC timestamps; encode the chosen local
            # calendar day explicitly rather than accidentally shifting it.
            from datetime import timezone
            stamp=int(datetime(anchor.year,anchor.month,anchor.day,tzinfo=timezone.utc).timestamp()*1000)
            norm=TemporalNormalizer.normalize(lower,stamp)
            if not norm['start']: return None
            start,end=date.fromisoformat(norm['start']),date.fromisoformat(norm['end'])
        return QueryTime(text,start,end)
    except (ValueError,OverflowError,OSError,ZoneInfoNotFoundError): return None


def expanded_query(query, constraint):
    if constraint is None: return query
    return query+'\nTime constraint for "'+constraint.expression+'": '+constraint.description()+'.'
