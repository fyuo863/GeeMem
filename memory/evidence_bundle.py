"""Assemble reviewed raw sources before Top-K; no generated factual text."""
import hashlib
import json


class EvidenceBundler:
    def __init__(self, cfg):
        self.mode = cfg.get('RAG_EVIDENCE_BUNDLE_MODE', 'off')
        if self.mode not in ('off', 'on'):
            raise ValueError('Invalid evidence bundle mode')
        self.max_sources = 4
        self.max_chars = 6000

    def assemble(self, user_id, ranked, supports, sufficient, top_k, trace):
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
