"""Conservative source time normalization and bounded temporal tie-breaking.

Normalized dates describe mentions, never inferred event facts. Message times
anchor explicit relative expressions only; they are not event occurrence dates.
"""
import re
from datetime import datetime, timedelta, timezone
import calendar

MONTHS={name.lower():i for i in range(1,13) for name in (calendar.month_name[i],calendar.month_abbr[i])}
MONTHS['sept']=9
MONTH_PATTERN='(?:'+'|'.join(sorted(MONTHS,key=len,reverse=True))+r')\.?'
ENGLISH_DATE=(r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+'+MONTH_PATTERN+r',?\s+\d{4}|'
              +MONTH_PATTERN+r'\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|'
              +MONTH_PATTERN+r'\s+\d{4})\b')


class TemporalNormalizer:
    VERSION = 'time-mentions-v2'
    PATTERN = re.compile(
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|'
        +ENGLISH_DATE+r'|'
        r'\b(?:yesterday|today|tomorrow|last month|next month|last year|next year)\b|'
        r'\b\d+ (?:days?|weeks?) ago\b|'
        r'\d{4}年\d{1,2}月\d{1,2}日|昨天|今天|明天|上个月|下个月', re.I)

    @staticmethod
    def normalize(text, timestamp=None):
        text = text.strip().lower()
        anchor = (datetime.fromtimestamp(timestamp / 1000, timezone.utc).date()
                  if timestamp is not None else None)
        try:
            numeric = re.fullmatch(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?', text)
            if numeric:
                day = datetime(*map(int, numeric.groups())).date()
                return dict(start=day.isoformat(), end=day.isoformat(), precision='day', anchored=False)
            reverse = re.fullmatch(r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?,?\s+(\d{4})',text)
            if reverse:
                text=f'{reverse[2]} {reverse[1]} {reverse[3]}'
            english = re.fullmatch(r'([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?', text)
            if english and english[3]:
                month=MONTHS.get(english[1])
                if month is None: raise ValueError('Unknown month')
                day = datetime(int(english[3]),month,int(english[2])).date()
                return dict(start=day.isoformat(), end=day.isoformat(), precision='day', anchored=False)
            month_year=re.fullmatch(r'([a-z]+)\.?\s+(\d{4})',text)
            if month_year and month_year[1] in MONTHS:
                month=MONTHS[month_year[1]];year=int(month_year[2])
                start=datetime(year,month,1).date()
                end=datetime(year,month,calendar.monthrange(year,month)[1]).date()
                return dict(start=start.isoformat(),end=end.isoformat(),precision='month',anchored=False)
            if anchor is not None:
                offsets = {'yesterday':-1, '昨天':-1, 'today':0, '今天':0, 'tomorrow':1, '明天':1}
                relative = re.fullmatch(r'(\d+) (days?|weeks?) ago', text)
                if text in offsets or relative:
                    days = offsets[text] if text in offsets else -int(relative[1]) * (7 if relative[2].startswith('week') else 1)
                    day = anchor + timedelta(days=days)
                    return dict(start=day.isoformat(), end=day.isoformat(), precision='day', anchored=True)
                if text in ('last month','next month','上个月','下个月'):
                    offset = -1 if text in ('last month','上个月') else 1
                    serial = anchor.year * 12 + anchor.month - 1 + offset
                    start = datetime(serial // 12, serial % 12 + 1, 1).date()
                    end = datetime((serial + 1) // 12, (serial + 1) % 12 + 1, 1).date() - timedelta(days=1)
                    return dict(start=start.isoformat(), end=end.isoformat(), precision='month', anchored=True)
                if text in ('last year','next year'):
                    year = anchor.year + (-1 if text == 'last year' else 1)
                    return dict(start=f'{year:04d}-01-01', end=f'{year:04d}-12-31', precision='year', anchored=True)
        except (ValueError, OverflowError, OSError):
            pass
        return dict(start=None, end=None, precision='unknown', anchored=False)

    @classmethod
    def mentions(cls, content, timestamp=None):
        return [dict(text=m.group(), char_start=m.start(), char_end=m.end(),
                     **cls.normalize(m.group(), timestamp)) for m in cls.PATTERN.finditer(content)]


class TemporalRanker:
    QUERY = re.compile(r'\b(when|before|after|between|ago|since|until|earliest|latest|first|last|order|sequence|days?|weeks?|months?|years?|January|February|March|April|May|June|July|August|September|October|November|December)\b|何时|什么时候|多久|几天|顺序|之前|之后|最早|最后|最近|\d{4}[-/年]', re.I)
    LATEST = re.compile(r'\b(latest|most recent|last (?:trip|visit|event|time))\b|最后|最近一次', re.I)
    EARLIEST = re.compile(r'\b(first|earliest)\b|最早|第一次', re.I)
    ORDER = re.compile(r'\b(order|sequence|before|after|between)\b|顺序|之前|之后', re.I)

    @classmethod
    def operator(cls, query):
        # Ordered/comparison questions need both endpoints, not one extreme.
        if cls.ORDER.search(query): return 'ordered'
        if cls.LATEST.search(query): return 'latest'
        if cls.EARLIEST.search(query): return 'earliest'
        return None

    @classmethod
    def order(cls, query, rows, indices, scores=None, *, tolerance=0.035, max_displacement=2):
        current = list(indices)
        op = cls.operator(query)
        if scores is None or op not in ('latest','earliest') or len(current) < 2:
            return current
        def date(i):
            row = dict(rows[i])
            mentions = row.get('time_mentions', [])
            # Multiple dates or imprecise intervals are ambiguous: do not infer
            # which event the question refers to or sort by message time.
            if len(mentions) == 1 and mentions[0]['precision'] == 'day':
                return mentions[0]['start']
            return None
        dates = {i: date(i) for i in current}
        values = [float(scores[i]) for i in current]
        span = max(values) - min(values)
        if span <= 0: return current
        original = {i:p for p,i in enumerate(current)}
        for p in range(1, min(len(current), 50)):
            j = p
            while j > 0:
                a,b = current[j],current[j-1]
                if (not dates[a] or not dates[b] or abs(float(scores[a])-float(scores[b])) > tolerance*span
                    or abs(original[a]-(j-1)) > max_displacement or abs(original[b]-j) > max_displacement):
                    break
                if not (dates[a]>dates[b] if op=='latest' else dates[a]<dates[b]): break
                current[j-1],current[j]=a,b
                j-=1
        return current
