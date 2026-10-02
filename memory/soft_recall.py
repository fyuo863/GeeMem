"""Bounded source-metadata recall; metadata never acts as a hard filter."""
import calendar
from datetime import datetime, timezone
import re


def query_date(query):
    iso=re.search(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',query)
    if iso:
        year,month,day=map(int,iso.groups())
        try:datetime(year,month,day)
        except ValueError:return None,None,None
        return year,month,day
    year_match=re.search(r'\b(19\d{2}|20\d{2}|21\d{2})\b',query)
    year=int(year_match[0]) if year_match else None
    for month in range(1,13):
        pattern=r'\b(?:'+calendar.month_name[month]+'|'+calendar.month_abbr[month]+r')\b'
        match=re.search(pattern,query,re.I)
        if not match:continue
        before=re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\s*$',query[:match.start()],re.I)
        after=re.match(r'\s+(\d{1,2})(?:st|nd|rd|th)?\b',query[match.end():],re.I)
        day=int((before or after)[1]) if before or after else None
        if day:
            try:datetime(year or 2000,month,day)
            except ValueError:return None,None,None
        return year,month,day
    return year,None,None


def metadata_candidates(query, rows, original_order, lexical_score, limit=60):
    from .provenance import metadata_text
    values=[dict(r) for r in rows]
    names={r.get('speaker') for r in values if r.get('speaker')}
    wanted={n for n in names if re.search(r'(?<!\w)'+re.escape(n)+r'(?!\w)',query,re.I)}
    year,month,day=query_date(query)
    if not wanted and year is None and month is None:return []
    lexical=lexical_score([metadata_text(r) for r in values],query)
    prior={i:r for r,i in enumerate(original_order,1)}
    eligible=[]
    for i,row in enumerate(values):
        person=row.get('speaker') in wanted
        stamp=row.get('timestamp') if row.get('timestamp') is not None else row.get('session_timestamp')
        date_match=False
        if stamp is not None and (year is not None or month is not None):
            date=datetime.fromtimestamp(stamp/1000,timezone.utc)
            date_match=(year is None or date.year==year) and (month is None or date.month==month) and (day is None or date.day==day)
        if person or date_match:eligible.append((i,int(person)+2*int(date_match)))
    # Date routes and speaker routes are complementary to the global route.
    # A record date is not asserted to be the date of every event in the text.
    return [i for i,_ in sorted(eligible,key=lambda x:(-x[1],-float(lexical[x[0]]),prior[x[0]]))[:limit]]
