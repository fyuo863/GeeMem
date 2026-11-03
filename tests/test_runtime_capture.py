"""Offline privacy, reassembly and failure-correlation checks; no live API calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('runtime_capture', Path(__file__).resolve().parents[1] / 'scripts/capture_runtime_errors.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class RuntimeCaptureTests(unittest.TestCase):
    def test_continuous_rotation_and_restart_append_preserve_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            log = capture.PrivateLog(directory, 400, retained_files=3, resume=True)
            for index in range(25):
                log.emit('sample', index=index, value='x'*30)
            self.assertFalse(log.full)
            log.close()
            paths = list(Path(directory).glob('events*.jsonl'))
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(p.stat().st_size <= 400 for p in paths))
            before = Path(directory, 'events.jsonl').read_bytes()
            log = capture.PrivateLog(directory, 400, retained_files=3, resume=True)
            self.assertEqual(log.used, len(before))
            log.emit('restart')
            log.close()
            records = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
            self.assertTrue(any(r.get('index') == 24 for r in records))
            self.assertTrue(any(r['event'] == 'restart' for r in records))

    def test_provider_retry_stream_and_error_allowlist(self):
        for name in ('embedding_retry', 'rerank_retry'):
            self.assertEqual(capture.safe_journal(name+' attempt=1 reason=ReadTimeout private=secret'),
                             {'kind':name, 'attempt':1, 'reason':'ReadTimeout'})
        safe = capture.safe_journal('WARNING llm_stream_failed headers_ms=1200 elapsed_ms=62000 '
                                    'error_type=ReadTimeout body=private-original')
        self.assertEqual(safe, {'kind':'llm_stream_failed', 'headers_ms':1200,
                               'elapsed_ms':62000, 'error_type':'ReadTimeout'})
        self.assertEqual(capture.safe_journal('embedding_upstream_error status=429 code=Arrearage inputs=2'),
                         {'kind':'embedding_upstream_error', 'status':429, 'code':'Arrearage', 'inputs':2})
        self.assertEqual(capture.safe_journal('embedding_retry attempt=2 reason=private-token')['reason'], 'other')
        self.assertIsNone(capture.safe_journal('unrelated private-original'))
        self.assertIsNone(capture.safe_journal('request_failed []'))

    def test_incomplete_request_and_chunked_response_have_connection_and_header_events(self):
        with tempfile.TemporaryDirectory() as directory:
            log = capture.PrivateLog(directory, 100000)
            observer = capture.APICapture(log)
            incoming = ('1.2.3.4',12345,'5.6.7.8',18092)
            outgoing = ('5.6.7.8',18092,'1.2.3.4',12345)
            observer.packet((*incoming,99,2,b''),18092,1)
            observer.packet((*outgoing,299,18,b''),18092,2)
            wire = b'POST /add?private-token HTTP/1.1\r\nAuthorization: private-token\r\nContent-Length: 999\r\n\r\nx'
            observer.packet((*incoming,100,24,wire),18092,3)
            reply = b'HTTP/1.1 502 Bad Gateway\r\nTransfer-Encoding: chunked\r\n\r\n'
            observer.packet((*outgoing,300,24,reply),18092,4)
            observer.packet((*outgoing,300+len(reply),4,b''),18092,5)
            log.close()
            text = Path(directory,'events.jsonl').read_text()
            self.assertNotIn('private-token', text)
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertTrue(any(r['event']=='api_tcp_open' for r in rows))
            self.assertTrue(any(r['event']=='api_tcp_accept' for r in rows))
            self.assertTrue(any(r['event']=='api_http_headers' and r.get('endpoint')=='/add' for r in rows))
            self.assertTrue(any(r['event']=='api_http_headers' and r.get('status')==502 for r in rows))
            self.assertTrue(any(r['event']=='api_tcp_close' and r['rst'] for r in rows))

    def test_fragmented_out_of_order_and_retransmitted_http(self):
        stream = capture.HTTPStream(4096)
        body = b'{"content":"original"}'
        request = b'POST /add HTTP/1.1\r\nAuthorization: Bearer private-token\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body
        self.assertEqual(stream.feed(99, b'', syn=True), [])
        self.assertEqual(stream.feed(130, request[30:]), [])
        self.assertEqual(stream.feed(100, request[:30]), [('POST /add HTTP/1.1', body)])
        self.assertEqual(stream.feed(100, request), [])

    def test_only_failed_add_body_saved_and_headers_never_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            log = capture.PrivateLog(directory, 100000)
            observer = capture.APICapture(log, save_failed=True)
            payload = {'user_id': 'private-user', 'request_id': 'private-request', 'messages': [{'content': 'private-original'}]}
            body = json.dumps(payload).encode()
            wire = b'POST /add HTTP/1.1\r\nAuthorization: Bearer private-token\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body
            def request(client_port, status):
                observer.packet(('1.2.3.4', client_port, '5.6.7.8', 18092, 100, 24, wire), 18092, 10)
                reply = f'HTTP/1.1 {status} Error\r\nContent-Length: 2\r\n\r\n{{}}'.encode()
                observer.packet(('5.6.7.8', 18092, '1.2.3.4', client_port, 300, 24, reply), 18092, 28)
            request(12345, 200)
            self.assertEqual(list(Path(directory).glob('failed-add-*')), [])
            request(12346, 502)
            request(12347, 502)
            log.close()
            saved = list(Path(directory).glob('failed-add-*'))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text()), payload)
            text = Path(directory, 'events.jsonl').read_text()
            self.assertNotIn('private-', text)
            self.assertNotIn('Authorization', text)
            rows = [json.loads(x) for x in text.splitlines()]
            self.assertTrue(any(r.get('status') == 502 and r['elapsed_ms'] == 18000 for r in rows))
            self.assertTrue(any(r.get('request_hash') == capture.identity('/add', payload) for r in rows))

    def test_budget_caps_body_and_log_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            log = capture.PrivateLog(directory, 256)
            self.assertIsNone(log.save_body(b'x'*300))
            log.emit('example', safe=True)
            log.emit('too_large', safe='x'*300)
            self.assertTrue(log.full)
            log.close()
            self.assertLessEqual(Path(directory, 'events.jsonl').stat().st_size, 256)
            with self.assertRaises(FileExistsError):
                capture.PrivateLog(directory, 1000)
        stream = capture.HTTPStream(100)
        self.assertEqual(stream.feed(0, b'POST /add HTTP/1.1\r\nContent-Length: 200\r\n\r\n'), [])
        self.assertTrue(stream.disabled)

    def test_journal_allowlist_and_chunked_capture_limit(self):
        row = {'endpoint': '/add', 'status': 502, 'stage': 'write_classifier',
               'exception': 'private-token', 'body': 'private-original',
               'causes': [{'type': 'RemoteProtocolError', 'message': 'private-token', 'location': 'llm.py:42'}]}
        safe = capture.safe_journal('request_failed '+json.dumps(row))
        self.assertEqual(safe['causes'], [{'type': 'RemoteProtocolError', 'location': 'llm.py:42'}])
        self.assertNotIn('private-', json.dumps(safe))
        stream = capture.HTTPStream(4096)
        self.assertEqual(stream.feed(0, b'POST /add HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\r\n0\r\n\r\n'), [])
        self.assertTrue(stream.disabled)

    def test_application_failure_saves_when_wire_response_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            log = capture.PrivateLog(directory, 100000)
            observer = capture.APICapture(log, save_failed=True)
            body = json.dumps({'user_id':'u','request_id':'r','messages':[{'content':'private-original'}]}).encode()
            payload = json.loads(body)
            import time
            now = time.monotonic()
            record = dict(body=body, request_hash=capture.identity('/add',payload),
                          body_sha256=capture.hashlib.sha256(body).hexdigest())
            observer.remember_body(record, 'connection', now)
            observer.application_failure(dict(endpoint='/add', request_hash=record['request_hash'],
                                              kind='request_finished',status=502))
            self.assertEqual(list(Path(directory).glob('failed-add-*')), [])
            observer.application_failure(dict(endpoint='/add', request_hash=record['request_hash'],
                                              kind='request_failed',status=502))
            self.assertEqual(len(list(Path(directory).glob('failed-add-*'))), 1)
            log.close()
            self.assertNotIn('private-original', Path(directory,'events.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
