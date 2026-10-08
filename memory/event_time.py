"""Bounded calendar parsing. Timestamps use the input adapter's UTC convention."""
import re
import calendar
from datetime import datetime, timezone, timedelta


def parse_time(expression, reference_timestamp):
    if not expression or expression.casefold() in ('null','unknown','none'):
        return None,None,'unknown'
    text=expression.strip().casefold()
    # Dates require a valid calendar date; no model-created boundaries are used.
    for fmt in ('%Y-%m-%d','%Y/%m/%d','%Y年%m月%d日','%d %B %Y','%B %d, %Y'):
        try:
            day=datetime.strptime(text,fmt).date().isoformat()
            return day,day,'day'
        except ValueError: pass
    if re.fullmatch(r'\d{4}',text) and 1<=int(text)<=9999:
        return text,text,'year'
    if reference_timestamp is None:
        return None,None,'relative'
    ref=datetime.fromtimestamp(reference_timestamp/1000,timezone.utc).date()
    years={'去年':-1,'last year':-1,'今年':0,'this year':0,'明年':1,'next year':1}
    months={'上个月':-1,'last month':-1,'本月':0,'this month':0,'下个月':1,'next month':1}
    days={'昨天':-1,'yesterday':-1,'今天':0,'today':0,'明天':1,'tomorrow':1}
    if text in years:
        year=str(ref.year+years[text]); return year,year,'year'
    if text in months:
        year,month=divmod(ref.year*12+ref.month-1+months[text],12)
        value=f'{year:04d}-{month+1:02d}';return value,value,'month'
    if text in days:
        day=(ref+timedelta(days=days[text])).isoformat();return day,day,'day'
    if text in ('last week','上周','this week','本周','next week','下周'):
        offset=-7 if text in ('last week','上周') else 7 if text in ('next week','下周') else 0
        start=ref-timedelta(days=ref.weekday())+timedelta(days=offset)
        return start.isoformat(),(start+timedelta(days=6)).isoformat(),'day'
    match=re.fullmatch(r'(\d+|one|two|three|four) (days?|weeks?) ago',text)
    if match:
        numbers={'one':1,'two':2,'three':3,'four':4}
        count=int(match[1]) if match[1].isdigit() else numbers[match[1]]
        day=(ref-timedelta(days=count*(7 if match[2].startswith('week') else 1))).isoformat()
        return day,day,'day'
    return None,None,'relative'
