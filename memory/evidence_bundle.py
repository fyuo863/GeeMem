"""Assemble reviewed raw sources before Top-K; no generated factual text."""
import hashlib
import json


class EvidenceBundler:
    def __init__(self, cfg):
        self.mode = cfg.get('RAG_EVIDENCE_BUNDLE_MODE', 'off')
        if self.mode not in ('off', 'on', 'raw'):
            raise ValueError('Invalid evidence bundle mode')
        self.max_sources = 4
        self.max_chars = 6000
        self.coverage = cfg.get('RAG_RAW_COVERAGE_MODE','off') == 'on'

    def assemble(self, user_id, ranked, supports, sufficient, top_k, trace):
        if self.mode == 'raw':
            return self.assemble_raw(user_id, ranked, supports, top_k, trace)
        base = {'data': ranked[:top_k]}
        stats = dict(status='skipped', sources=0, chars=0)
        trace['evidence_bundle'] = stats
        audit = trace.get('evidence_chain', {})
        chain_nodes = audit.get('nodes', [])
        def finish(result, source_ids):
            if audit:
                expected = {n['source_id'] for n in chain_nodes}
                audit['returned_source_ids'] = list(source_ids)
                audit['omitted_source_ids'] = sorted(expected - set(source_ids))
                audit['delivery_complete'] = bool(expected) and not audit['omitted_source_ids']
            return result
        if self.mode != 'on' or (not sufficient and not chain_nodes):
            return finish(base, [h['id'] for h in base['data']])
        chain_order = {n['source_id']: i for i, n in enumerate(chain_nodes)}
        members = [h for h in ranked if h['id'] in supports]
        if chain_order:
            members.sort(key=lambda h: chain_order.get(h['id'], len(chain_order)))
        if not 2 <= len(members) <= self.max_sources:
            stats['status'] = 'source_limit'
            return finish(base, [h['id'] for h in base['data']])
        # Recheck even when upstream validation was configured to trust the LLM.
        if any(not supports[h['id']].quote or supports[h['id']].quote not in h['content'] for h in members):
            stats['status'] = 'invalid_quote'
            return finish(base, [h['id'] for h in base['data']])
        segments = []
        for i, h in enumerate(members):
            label = ('候选原文' if audit and audit.get('status') != 'complete'
                     else ('核心原文' if i == 0 else '关联原文'))
            node = chain_nodes[chain_order[h['id']]] if h['id'] in chain_order else None
            hop = f'; hop={node["hop"]}; derived_from={json.dumps(node["derived_from"], ensure_ascii=False)}' if node else ''
            segments.append(f'[{label} {i+1}{hop}; source_id={json.dumps(h["id"], ensure_ascii=False)}; '
                            f'source_time={json.dumps(h.get("created_at"), ensure_ascii=False)}]\n{h["content"]}')
        content = '\n\n'.join(segments)
        if audit:
            content = '[Evidence set status: '+audit['status']+'; source-grounded, semantic sufficiency is model-assessed]\n'+content
        if len(content) > self.max_chars:
            stats['status'] = 'char_limit'
            return finish(base, [h['id'] for h in base['data']])
        member_ids = [h['id'] for h in members]
        # Source order and full content are part of identity; never reuse one raw ID.
        identity = json.dumps(['bundle-v1', user_id, content], ensure_ascii=False, separators=(',', ':'))
        bundle_id = 'bundle_' + hashlib.sha256(identity.encode('utf8')).hexdigest()
        bundle = dict(id=bundle_id, content=content, score=members[0]['score'])
        # No single source timestamp can represent the whole bundle.
        result=[]; inserted=False
        for h in ranked:
            if h['id'] in member_ids:
                if not inserted:
                    result.append(bundle); inserted=True
            else:
                result.append(h)
        stats.update(status='assembled', sources=len(members), chars=len(content))
        returned = result[:top_k]
        bundle_visible = any(h['id'] == bundle_id for h in returned)
        return finish({'data': returned, '_bundles': [dict(id=bundle_id, source_ids=member_ids)] if bundle_visible else []},
                      (member_ids if bundle_visible else []) + [h['id'] for h in returned if h['id'] != bundle_id])

    def assemble_raw(self, user_id, ranked, supports, top_k, trace):
        # The candidate scope has already passed user/version/disclosure filters.
        # Only exact, program-bound source references can enter a package.
        unique = {}
        for hit in ranked:
            unique.setdefault(hit['id'], hit)
        ranked = list(unique.values())
        selected = [h for h in ranked if h['id'] in supports
                   and supports[h['id']].quote
                   and supports[h['id']].quote in h['content']]
        members = selected if self.coverage else selected[:8]
        # Timestamp order is presentation only; no earlier source is invalidated.
        members.sort(key=lambda h: (h.get('created_at') is None, h.get('created_at') or ''))
        header = ('[Related original sources; relationships, current validity and event '
                  'stages are not adjudicated. source_time is message time, not event time.]\n')
        groups = []; group = []; size = len(header)
        for h in members:
            segment = (f'[source_id={json.dumps(h["id"], ensure_ascii=False)}; '
                       f'source_time={json.dumps(h.get("created_at"), ensure_ascii=False)}]\n{h["content"]}')
            if len(header) + len(segment) > self.max_chars:
                continue  # Retain the full oversized source as an ordinary result.
            if group and (len(group) == self.max_sources or size + len(segment) + 2 > self.max_chars):
                groups.append(group); group=[]; size=len(header)
            group.append((h, segment)); size += len(segment) + 2
        if group: groups.append(group)
        replacements = {}; consumed = set(); mappings = []
        for group in groups:
            if len(group) < 2: continue
            content = header + '\n\n'.join(segment for _,segment in group)
            identity = json.dumps(['raw-bundle-v1', user_id, content], ensure_ascii=False)
            bid = 'bundle_' + hashlib.sha256(identity.encode('utf8')).hexdigest()
            ids = [h['id'] for h,_ in group]
            first = next(h for h in ranked if h['id'] in ids)
            replacements[first['id']] = dict(id=bid, content=content, score=first['score'])
            consumed.update(ids)
            mappings.append(dict(id=bid, source_ids=ids))
        eligible = [replacements[h['id']] if h['id'] in replacements else h for h in ranked
                    if h['id'] in replacements or h['id'] not in consumed]
        if self.coverage:
            priority={h['id'] for h in selected} | {b['id'] for b in mappings}
            eligible.sort(key=lambda h:h['id'] not in priority)
        result = eligible[:top_k]
        returned = {h['id'] for h in result}
        mappings = [b for b in mappings if b['id'] in returned]
        delivered = returned | {i for b in mappings for i in b['source_ids']}
        trace['evidence_bundle'] = dict(status='assembled' if mappings else 'raw_sources',
            mode='raw', semantic_status='not_adjudicated', packages=len(mappings),
            selected_source_ids=[h['id'] for h in selected],
            omitted_source_ids=[h['id'] for h in selected if h['id'] not in delivered])
        return dict(data=result, _bundles=mappings)
