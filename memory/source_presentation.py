"""Present source timestamps beside originals, only after retrieval is complete."""
from datetime import datetime
import json


def present_sources(result, bundles, trace):
    """No date inference, text rewriting, ranking changes or extra model calls.

    Bundles already carry a timestamp per member. Never assign a single date to
    their combined content. Only timezone-qualified source dates are displayed;
    unknown dates must not be replaced by the query clock or the server clock.
    """
    bundle_ids = {b['id'] for b in bundles}
    presented = []; anchored = []; unknown = []
    for hit in result['data']:
        if hit['id'] in bundle_ids:
            presented.append(hit)
            continue
        timestamp = hit.get('created_at')
        try:
            parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            # Competition timestamp=0 denotes unknown, not a January 1970 anchor.
            valid = parsed.utcoffset() is not None and parsed.timestamp() != 0
        except (TypeError, AttributeError, ValueError):
            valid = False
        if not valid:
            unknown.append(hit['id'])
            presented.append(hit)
            continue
        header = (f'[Original source; source_id={json.dumps(hit["id"], ensure_ascii=False)}; '
                  f'source_time={json.dumps(timestamp, ensure_ascii=False)}; '
                  'source_time is message time, not event time. '
                  'Relative dates in the original may use this message time as an anchor; '
                  'quoted or explicitly dated passages may use a different anchor.]\n')
        presented.append(dict(hit, content=header+hit['content']))
        anchored.append(hit['id'])
    trace['source_presentation'] = dict(anchored_source_ids=anchored,
        unknown_time_source_ids=unknown, bundle_ids=sorted(bundle_ids))
    return dict(result, data=presented)
