"""Receive complete structured completions over SSE without exposing partial data."""
import json
import logging
import time

import httpx


_log = logging.getLogger(__name__)
_MAX_WIRE_BYTES = 8 * 1024 * 1024
_MAX_ELAPSED_SECONDS = 60


class StreamingJSONClient:
    """A post-only adapter preserving the existing completion response contract.

    The wrapped client's 60-second I/O timeout still applies. The elapsed guard
    is checked between received lines; it does not create another retry layer.
    """

    def __init__(self, client):
        self.client = client

    def post(self, url, *, json: dict, **kwargs):
        payload = dict(json, stream=True, stream_options={'include_usage': True})
        started = time.monotonic()
        with self.client.stream('POST', url, json=payload, **kwargs) as response:
            if response.is_error:
                response.read()
                return response
            headers_ms = round((time.monotonic() - started) * 1000)
            try:
                result = self._receive(response, started)
            except Exception as exc:
                _log.warning('llm_stream_failed headers_ms=%d elapsed_ms=%d error_type=%s',
                             headers_ms, round((time.monotonic()-started)*1000), type(exc).__name__)
                raise
            _log.warning('llm_stream_finished headers_ms=%d elapsed_ms=%d',
                         headers_ms, round((time.monotonic()-started)*1000))
            # Preserve provider headers used for tracing, but replace SSE framing.
            headers = {key: value for key, value in response.headers.items()
                       if key.lower() not in ('content-type', 'content-length',
                                              'content-encoding', 'transfer-encoding')}
            return httpx.Response(response.status_code, json=result, headers=headers,
                                  request=response.request)

    @staticmethod
    def _receive(response, started):
        content = []
        event_lines = []
        wire_bytes = 0
        finish_reason = None
        usage = None
        done = False
        metadata = {}

        def events():
            nonlocal wire_bytes
            for line in response.iter_lines():
                if time.monotonic()-started > _MAX_ELAPSED_SECONDS:
                    raise httpx.ReadTimeout('Structured stream exceeded elapsed budget',
                                            request=response.request)
                wire_bytes += len(line.encode('utf-8')) + 1
                if wire_bytes > _MAX_WIRE_BYTES:
                    raise httpx.RemoteProtocolError('Structured stream exceeds size limit',
                                                    request=response.request)
                if not line:
                    if event_lines:
                        data = '\n'.join(event_lines)
                        event_lines.clear()
                        yield data
                elif line.startswith('data:'):
                    event_lines.append(line[5:].removeprefix(' '))
            # Providers may omit the final blank line after the last SSE event.
            if event_lines:
                yield '\n'.join(event_lines)

        for data in events():
            if data == '[DONE]':
                done = True
                break
            chunk = json.loads(data)
            if not isinstance(chunk, dict) or 'error' in chunk:
                raise ValueError('Invalid structured stream event')
            for key in ('id', 'model', 'created'):
                if key in chunk:
                    metadata[key] = chunk[key]
            if isinstance(chunk.get('usage'), dict):
                usage = chunk['usage']
            choices = chunk.get('choices', [])
            if not isinstance(choices, list):
                raise ValueError('Invalid structured choices')
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError('Invalid structured choice')
                if choice.get('index', 0) != 0:
                    raise ValueError('Unexpected completion choice')
                delta = choice.get('delta', {})
                if not isinstance(delta, dict):
                    raise ValueError('Invalid structured delta')
                if delta.get('refusal'):
                    raise ValueError('Model refused structured completion')
                part = delta.get('content')
                if part is not None:
                    if not isinstance(part, str):
                        raise ValueError('Invalid structured content delta')
                    content.append(part)
                reason = choice.get('finish_reason')
                if reason:
                    finish_reason = reason
        if not done or finish_reason != 'stop' or not content:
            raise httpx.RemoteProtocolError('Structured stream did not finish normally',
                                            request=response.request)
        result = dict(metadata, choices=[{'index': 0, 'finish_reason': finish_reason,
                      'message': {'role': 'assistant', 'content': ''.join(content)}}])
        if usage is not None:
            result['usage'] = usage
        return result
