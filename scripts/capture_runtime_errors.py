"""Bounded Linux sidecar: observe HTTP/TCP and journal without changing the API.

No proxy, application hook, database connection, provider call, or packet injection.
Only failed Add JSON bodies may be saved; HTTP headers and TLS bytes are never saved.
Run as root (AF_PACKET), with a private output directory outside the repository.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import threading
import time


def identity(endpoint, payload):
    fields = [endpoint, payload.get('user_id'), payload.get('request_id', '')]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=True).encode()).hexdigest()[:24]


class PrivateLog:
    def __init__(self, directory, limit, retained_files=1, resume=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.directory, 0o700)
        self.limit = limit
        self.retained_files = retained_files
        self.used = 0
        self.lock = threading.Lock()
        self.full = False
        self.path = self.directory / 'events.jsonl'
        self.handle = self._open(resume)
        self.used = os.fstat(self.handle.fileno()).st_size

    def _open(self, resume=False):
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if resume else os.O_EXCL)
        flags |= getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(self.path, flags, 0o600)
        os.chmod(self.path, 0o600)
        return os.fdopen(fd, 'ab' if resume else 'wb')

    def _rotate(self):
        self.handle.close()
        for index in range(self.retained_files - 1, 0, -1):
            target = self.directory / ('events.' + str(index) + '.jsonl')
            source = self.path if index == 1 else self.directory / ('events.' + str(index-1) + '.jsonl')
            if source.exists():
                os.replace(source, target)
        self.handle = self._open()
        self.used = 0

    def emit(self, event, **fields):
        record = dict(ts=datetime.now(timezone.utc).isoformat(), event=event, **fields)
        data = (json.dumps(record, ensure_ascii=True, separators=(',', ':')) + '\n').encode()
        with self.lock:
            if len(data) > self.limit:
                self.full = True
                return
            if self.used + len(data) > self.limit:
                if self.retained_files == 1:
                    self.full = True
                    return
                self._rotate()
            self.handle.write(data)
            self.handle.flush()
            self.used += len(data)

    def save_body(self, body):
        digest = hashlib.sha256(body).hexdigest()
        name = 'failed-add-' + digest + '.json'
        target = self.directory / name
        with self.lock:
            if target.exists():
                return name
            if self.used + len(body) > self.limit:
                return None
            with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as f:
                f.write(body)
            self.used += len(body)
        return name

    def close(self):
        with self.lock:
            self.handle.close()


def decode_ipv4(frame):
    """Ethernet IPv4/TCP only; do not decode TLS or return unrelated packets."""
    if len(frame) < 54 or frame[12:14] != b'\x08\x00':
        return None
    ip = frame[14:]
    size = (ip[0] & 15) * 4
    total = struct.unpack('!H', ip[2:4])[0]
    if size < 20 or ip[9] != 6 or struct.unpack('!H', ip[6:8])[0] & 0x3fff:
        return None
    tcp = ip[size:total]
    if len(tcp) < 20:
        return None
    offset = (tcp[12] >> 4) * 4
    if offset < 20 or offset > len(tcp):
        return None
    source, dest, seq = struct.unpack('!HHI', tcp[:8])
    return (socket.inet_ntoa(ip[12:16]), source,
            socket.inet_ntoa(ip[16:20]), dest, seq, tcp[13], tcp[offset:])


class HTTPStream:
    """Reassemble bounded HTTP/1 content-length messages, including retransmits."""
    def __init__(self, limit):
        self.limit = limit
        self.seq = None
        self.pending = {}
        self.pending_bytes = 0
        self.buffer = bytearray()
        self.disabled = False
        self.headers = []
        self.header_reported = False

    def feed(self, seq, data, syn=False):
        self.headers = []
        if self.disabled:
            return []
        if syn:
            self.seq = (seq + 1) & 0xffffffff
        if not data:
            return []
        if self.seq is None:
            # Capture may start mid-connection. Never treat arbitrary body bytes as a header.
            if not data.startswith((b'POST ', b'GET ', b'HTTP/1.')):
                return []
            self.seq = seq
        if seq + len(data) <= self.seq:
            return []
        if seq < self.seq:
            data = data[self.seq-seq:]
            seq = self.seq
        if seq not in self.pending:
            self.pending[seq] = data
            self.pending_bytes += len(data)
        if len(self.buffer) + self.pending_bytes > self.limit:
            self.disabled = True
            self.buffer.clear()
            self.pending.clear()
            self.pending_bytes = 0
            return []
        while self.pending:
            start = min(self.pending)
            chunk = self.pending[start]
            if start > self.seq:
                break
            self.pending.pop(start)
            self.pending_bytes -= len(chunk)
            skip = self.seq-start
            if skip < len(chunk):
                self.buffer.extend(chunk[skip:])
                self.seq += len(chunk)-skip
        messages = []
        while b'\r\n\r\n' in self.buffer:
            head_end = self.buffer.index(b'\r\n\r\n')
            if head_end > 32768:
                self.disabled = True
                break
            lines = bytes(self.buffer[:head_end]).split(b'\r\n')
            if not self.header_reported:
                self.headers.append(lines[0].decode('ascii', errors='replace'))
                self.header_reported = True
            length = 0
            for line in lines[1:]:
                key, _, value = line.partition(b':')
                if key.lower() == b'transfer-encoding' and value.strip().lower() != b'identity':
                    # Explicitly fail capture, never guess boundaries or affect live traffic.
                    self.disabled = True
                    break
                if key.lower() == b'content-length':
                    try:
                        length = int(value.strip())
                    except ValueError:
                        self.disabled = True
                        break
            if self.disabled or length < 0 or length + head_end + 4 > self.limit:
                self.disabled = True
                break
            end = head_end + 4 + length
            if len(self.buffer) < end:
                break
            # Headers, including Authorization, are discarded here.
            messages.append((lines[0].decode('ascii', errors='replace'), bytes(self.buffer[head_end+4:end])))
            del self.buffer[:end]
            self.header_reported = False
        if self.disabled:
            self.buffer.clear()
            self.pending.clear()
            self.pending_bytes = 0
        return messages


class APICapture:
    def __init__(self, log, save_failed=False, body_limit=8*1024*1024, max_connections=16):
        self.log, self.save_failed = log, save_failed
        self.body_limit, self.max_connections = body_limit, max_connections
        self.connections = {}
        self.body_lock = threading.Lock()
        self.failed_candidates = {}

    def remember_body(self, record, connection, now):
        if record.get('body') is None:
            return
        key = (record['request_hash'], record['body_sha256'])
        with self.body_lock:
            self.failed_candidates[key] = dict(body=record['body'], connection=connection, seen=now)
            # Keep bounded candidates when response framing is unavailable. A hash
            # alone identifies the request ID, not an identical request body.
            while (len(self.failed_candidates) > 16 or
                   sum(len(v['body']) for v in self.failed_candidates.values()) > 16*1024*1024):
                oldest = min(self.failed_candidates, key=lambda k: self.failed_candidates[k]['seen'])
                self.failed_candidates.pop(oldest)

    def application_failure(self, diagnostic):
        if (not self.save_failed or diagnostic.get('kind') != 'request_failed'
                or diagnostic.get('endpoint') != '/add' or diagnostic.get('status', 0) < 500):
            return
        now = time.monotonic()
        with self.body_lock:
            candidates = [(key, dict(value)) for key, value in self.failed_candidates.items()
                          if key[0] == diagnostic.get('request_hash') and now-value['seen'] <= 300]
        for (request_hash, body_hash), value in candidates:
            saved = self.log.save_body(value['body'])
            self.log.emit('failed_add_candidate', request_hash=request_hash, body_sha256=body_hash,
                          connection=value['connection'], payload_file=saved,
                          association='application_failure_request_hash_and_recent_body',
                          candidate_count=len(candidates))

    def packet(self, packet, port, now):
        src, sport, dst, dport, seq, flags, data = packet
        incoming = dport == port
        key = (src, sport, dst, dport) if incoming else (dst, dport, src, sport)
        # A new SYN means a new connection even if the port tuple was reused.
        if incoming and flags & 2 and not flags & 16:
            self.connections.pop(key, None)
        if key not in self.connections:
            if len(self.connections) >= self.max_connections:
                oldest = min(self.connections, key=lambda k: self.connections[k]['seen'])
                self.log.emit('capture_evicted', connection=self.connections[oldest]['id'])
                self.connections.pop(oldest)
            self.connections[key] = dict(id=hashlib.sha256(repr((key, now)).encode()).hexdigest()[:16],
                streams=[HTTPStream(self.body_limit), HTTPStream(self.body_limit)], pending=deque(), seen=now)
        conn = self.connections[key]
        conn['seen'] = now
        if incoming and flags & 2 and not flags & 16:
            self.log.emit('api_tcp_open', connection=conn['id'], side='client')
        elif not incoming and flags & 2 and flags & 16:
            self.log.emit('api_tcp_accept', connection=conn['id'], side='server')
        stream = conn['streams'][0 if incoming else 1]
        was_disabled = stream.disabled
        for first, body in stream.feed(seq, data, syn=bool(flags & 2)):
            if incoming:
                parts = first.split()
                if len(parts) != 3:
                    continue
                endpoint = parts[1].split('?', 1)[0]
                record = dict(endpoint=endpoint if endpoint in ('/add', '/search', '/health') else 'other',
                              started=now, body=None)
                if endpoint in ('/add', '/search'):
                    try:
                        payload = json.loads(body)
                        record['request_hash'] = identity(endpoint, payload)
                        record['body_sha256'] = hashlib.sha256(body).hexdigest()
                        messages = payload.get('messages', [])
                        record['message_count'] = len(messages)
                        record['content_chars'] = sum(len(m.get('content', '')) for m in messages)
                        if self.save_failed and endpoint == '/add':
                            record['body'] = body
                    except (ValueError, TypeError, AttributeError):
                        record['parse_error'] = True
                # Keep at most two pipelined bodies; ordinary clients are sequential.
                if len(conn['pending']) >= 2:
                    conn['pending'].popleft()
                    self.log.emit('capture_pending_evicted', connection=conn['id'])
                conn['pending'].append(record)
                self.remember_body(record, conn['id'], now)
                self.log.emit('api_request', connection=conn['id'], body_bytes=len(body),
                              **{k: v for k, v in record.items() if k not in ('started', 'body')})
            elif first.startswith('HTTP/'):
                try:
                    status = int(first.split()[1])
                except (ValueError, IndexError):
                    continue
                if status < 200:
                    continue
                record = conn['pending'].popleft() if conn['pending'] else {}
                saved = None
                if status >= 500 and record.get('body') is not None:
                    saved = self.log.save_body(record['body'])
                self.log.emit('api_response', connection=conn['id'], status=status,
                    elapsed_ms=round((now-record['started'])*1000) if 'started' in record else None,
                    request_hash=record.get('request_hash'), payload_file=saved)
        # Record arrival/status even if the body is incomplete, chunked or oversized.
        for first in stream.headers:
            parts = first.split()
            fields = dict(connection=conn['id'], direction='request' if incoming else 'response')
            if incoming and len(parts) == 3:
                fields['method'] = parts[0] if parts[0] in ('POST', 'GET', 'HEAD', 'OPTIONS') else 'other'
                endpoint = parts[1].split('?', 1)[0]
                fields['endpoint'] = endpoint if endpoint in ('/add', '/search', '/health') else 'other'
            elif not incoming and len(parts) >= 2 and parts[0].startswith('HTTP/') and parts[1].isdigit():
                fields['status'] = int(parts[1])
            else:
                continue
            self.log.emit('api_http_headers', **fields)
        if stream.disabled and not was_disabled:
            self.log.emit('http_capture_unavailable', connection=conn['id'],
                          direction='request' if incoming else 'response',
                          reason='unsupported_framing_or_size_limit')
        if flags & 5:
            self.log.emit('api_tcp_close', connection=conn['id'],
                          side='client' if incoming else 'server', rst=bool(flags & 4),
                          pending_requests=len(conn['pending']),
                          request_hashes=[r['request_hash'] for r in conn['pending'] if 'request_hash' in r])
            if flags & 4:
                self.connections.pop(key, None)

    def expire(self, now):
        for key in list(self.connections):
            if now-self.connections[key]['seen'] > 300:
                self.log.emit('api_capture_expired', connection=self.connections[key]['id'],
                              pending_requests=len(self.connections[key]['pending']))
                self.connections.pop(key)
        with self.body_lock:
            self.failed_candidates = {k: v for k, v in self.failed_candidates.items() if now-v['seen'] <= 300}


def safe_journal(message):
    for marker in ('request_failed ', 'request_finished ', 'provider_started ', 'provider_finished '):
        if marker not in message:
            continue
        try:
            item = json.loads(message.split(marker, 1)[1])
        except (ValueError, TypeError):
            return None
        if not isinstance(item, dict):
            return None
        out = {k: item[k] for k in ('endpoint', 'request_hash', 'stage', 'stage_elapsed_ms', 'status',
                                  'elapsed_ms', 'call_id', 'provider_kind', 'model', 'error_type',
                                  'document_count', 'input_count', 'input_tokens', 'output_tokens') if k in item}
        out['kind'] = marker.strip()
        out['causes'] = [{k: row[k] for k in ('type', 'location', 'upstream_status', 'code') if k in row}
                         for row in item.get('causes', []) if isinstance(row, dict)]
        return out
    # Match named events and allowlisted values only: never copy arbitrary log text.
    match = re.search(r'\b(llm_stream_failed|llm_stream_finished|embedding_retry|rerank_retry|'
                      r'reranker_retry|embedding_upstream_error|rerank_upstream_error|'
                      r'embedding_query_bounded|rerank_query_bounded)\b', message)
    if match:
        out = {'kind': match.group(1)}
        numeric = {'headers_ms', 'elapsed_ms', 'attempt', 'status', 'inputs', 'documents',
                   'query_chars', 'max_document_chars', 'max_input_bytes', 'original_chars',
                   'sent_chars', 'original_bytes', 'sent_bytes'}
        reasons = {'ReadTimeout', 'ConnectTimeout', 'WriteTimeout', 'PoolTimeout',
                   'RemoteProtocolError', 'ConnectError', 'ReadError', 'WriteError',
                   'LLMStreamError', 'HTTPStatusError'}
        codes = {'InvalidParameter', 'Arrearage', 'Throttling', 'InvalidApiKey', 'other'}
        for key, value in re.findall(r'\b([a-z_]+)=([^\s]+)', message[match.end():]):
            if key in numeric and value.isdigit():
                out[key] = int(value)
            elif key in ('reason', 'error_type'):
                out[key] = value if value in reasons else 'other'
            elif key == 'code':
                out[key] = value if value in codes else 'other'
        return out
    if 'Invalid HTTP request received' in message:
        return {'kind': 'invalid_http_request'}
    return None


def journal_worker(unit, log, stop, on_failure):
    process = subprocess.Popen(['journalctl', '-u', unit, '-f', '-n', '0', '-o', 'json', '--no-pager'],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    pending = b''
    try:
        log.emit('journal_attached', unit=unit, pid=process.pid)
        while not stop.is_set():
            import select
            if not select.select([process.stdout], [], [], 0.5)[0]:
                continue
            data = os.read(process.stdout.fileno(), 65536)
            if not data:
                log.emit('journal_disconnected', unit=unit, returncode=process.poll())
                break
            pending += data
            while b'\n' in pending:
                line, pending = pending.split(b'\n', 1)
                try:
                    entry = json.loads(line)
                    safe = safe_journal(entry.get('MESSAGE', ''))
                    if safe:
                        log.emit('application_diagnostic', journal_us=entry.get('__REALTIME_TIMESTAMP'), **safe)
                        on_failure(safe)
                except (ValueError, TypeError):
                    pass
            if len(pending) > 1024*1024:
                pending = b''
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def proxy_connections(unit):
    """Read the proxy's own socket tuples; do not read its credential-bearing config."""
    try:
        pid = subprocess.check_output(['systemctl', 'show', unit, '-p', 'MainPID', '--value'], text=True, timeout=2).strip()
        inodes = set()
        for fd in Path('/proc', pid, 'fd').iterdir():
            try:
                target = os.readlink(fd)
                if target.startswith('socket:['):
                    inodes.add(target[8:-1])
            except OSError:
                pass
        found = set()
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            fields = line.split()
            if fields[9] not in inodes or fields[3] == '0A':
                continue
            ends = []
            for addr in fields[1:3]:
                ip, port = addr.split(':')
                ends.extend((socket.inet_ntoa(bytes.fromhex(ip)[::-1]), int(port, 16)))
            if ends[2] != '127.0.0.1':
                found.add(tuple(ends))
        return found
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return set()


def run(args):
    if not hasattr(socket, 'AF_PACKET'):
        raise SystemExit('Linux AF_PACKET is required; the API does not need a restart.')
    os.umask(0o077)
    log = PrivateLog(args.output, args.max_mb*1024*1024,
                     retained_files=args.rotate_files if args.continuous else 1, resume=args.continuous)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    api = APICapture(log, args.save_failed_add)
    # ETH_P_IP omits physical-interface outgoing packets on this Linux host.
    # Receive ETH_P_ALL so SYN-ACK/responses are visible; decode_ipv4 still filters.
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
    sock.settimeout(0.5)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)
    worker = threading.Thread(target=journal_worker, args=(args.service, log, stop, api.application_failure), daemon=True)
    worker.start()
    started = time.monotonic()
    next_scan = 0
    upstream = set()
    flows = {}
    packets = 0
    next_heartbeat = started
    log.emit('capture_started', pid=os.getpid(), api_port=args.api_port, proxy_port=args.proxy_port,
             duration_seconds=None if args.continuous else args.duration, save_failed_add=args.save_failed_add,
             retained_files=args.rotate_files if args.continuous else 1,
             capture_protocol='IPv4 TCP; HTTP/1 content-length; TLS metadata only')
    try:
        while not stop.is_set() and not log.full and (args.continuous or time.monotonic()-started < args.duration):
            now = time.monotonic()
            if now >= next_heartbeat:
                stats = None
                try:
                    received, dropped = struct.unpack('II', sock.getsockopt(263, 6, 8))
                    stats = dict(socket_packets=received, socket_drops=dropped)
                except OSError:
                    pass
                log.emit('capture_heartbeat', elapsed_seconds=round(now-started, 1),
                         journal_alive=worker.is_alive(), api_connections=len(api.connections),
                         observed_proxy_packets=packets, packet_statistics=stats)
                if not worker.is_alive():
                    raise RuntimeError('Journal reader stopped; restart the diagnostic sidecar')
                next_heartbeat = now+args.heartbeat
            if now >= next_scan:
                upstream = proxy_connections(args.proxy_service)
                next_scan = now+2
                api.expire(now)
                flows = {k: v for k, v in flows.items() if now-v['last'] < 120}
            try:
                frame, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            # Loopback has both HOST and OUTGOING copies; ethernet needs both directions.
            if addr[0] == 'lo' and addr[2] == socket.PACKET_OUTGOING:
                continue
            now = time.monotonic()
            packet = decode_ipv4(frame)
            if packet is None:
                continue
            src, sport, dst, dport, seq, flags, data = packet
            if args.api_port in (sport, dport):
                api.packet(packet, args.api_port, now)
            local_proxy = args.proxy_port in (sport, dport) and src == dst == '127.0.0.1'
            key = (src, sport, dst, dport)
            outer = key in upstream or (dst, dport, src, sport) in upstream
            if not local_proxy and not outer:
                continue
            if not data and not flags & 7:
                continue
            if local_proxy:
                key = key if dport == args.proxy_port else (dst, dport, src, sport)
                side = 'application' if dport == args.proxy_port else 'proxy'
            else:
                outgoing = key in upstream
                key = key if outgoing else (dst, dport, src, sport)
                side = 'local_proxy' if outgoing else 'remote_peer'
            if flags & 2 and not flags & 16:
                flows.pop(key, None)
            state = flows.setdefault(key, {'start': now, 'last': now, 'id': hashlib.sha256(repr((key, now)).encode()).hexdigest()[:16]})
            quiet = round((now-state['last'])*1000)
            state['last'] = now
            packets += 1
            fields = dict(connection=state['id'], layer='application_proxy' if local_proxy else 'proxy_upstream',
                          side=side, local_port=key[1], peer_port=key[3], bytes=len(data),
                          since_start_ms=round((now-state['start'])*1000), quiet_ms=quiet,
                          seq=seq, syn=bool(flags & 2), fin=bool(flags & 1), rst=bool(flags & 4))
            if local_proxy and data.startswith(b'HTTP/1.'):
                try:
                    fields['connect_status'] = int(data.split(b' ', 2)[1])
                except (ValueError, IndexError):
                    pass
            if local_proxy and data and data[0] in (20, 21, 22, 23) and len(data) >= 5 and data[1] == 3:
                fields['tls_record_type'] = data[0]
            log.emit('tcp_observation', **fields)
        drops = None
        try:
            received, dropped = struct.unpack('II', sock.getsockopt(263, 6, 8))
            drops = dict(socket_packets=received, socket_drops=dropped)
        except OSError:
            pass
        log.emit('capture_stopped', elapsed_seconds=round(time.monotonic()-started, 1),
                 observed_proxy_packets=packets, packet_statistics=drops, size_limit=log.full)
    finally:
        sock.close()
        stop.set()
        worker.join(timeout=4)
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=int, default=7200)
    parser.add_argument('--max-mb', type=int, default=64)
    parser.add_argument('--api-port', type=int, default=18092)
    parser.add_argument('--proxy-port', type=int, default=18093)
    parser.add_argument('--service', default='csig-aml.service')
    parser.add_argument('--proxy-service', default='csig-aml-proxy.service')
    parser.add_argument('--save-failed-add', action='store_true')
    parser.add_argument('--continuous', action='store_true', help='Run until stopped; append and rotate private logs')
    parser.add_argument('--rotate-files', type=int, default=4, help='Total event files retained in continuous mode')
    parser.add_argument('--heartbeat', type=int, default=30, help='Diagnostic heartbeat interval in seconds')
    args = parser.parse_args()
    if not 1 <= args.duration <= 86400 or not 1 <= args.max_mb <= 256:
        parser.error('duration must be 1..86400 seconds and max-mb 1..256')
    if not 2 <= args.rotate_files <= 16 or not 5 <= args.heartbeat <= 300:
        parser.error('rotate-files must be 2..16 and heartbeat must be 5..300 seconds')
    if args.continuous and args.save_failed_add:
        parser.error('continuous mode records metadata only; failed original bodies require a bounded separate capture')
    run(args)


if __name__ == '__main__':
    main()
