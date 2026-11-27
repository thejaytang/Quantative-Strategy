#!/usr/bin/env python3
"""Precise static check for real broker / live-order APIs in a strategy file.

has_broker is True when the file talks to a broker, a live or paper trading
API, submits orders to an external venue, opens a socket/websocket, issues
an HTTP POST/PUT/DELETE, contacts a data host outside the allowlist
(Yahoo/yfinance and FRED/ALFRED only, plus exact GET-only hosts approved
one-by-one by Gatekeeper), starts a subprocess / os.system / os.popen,
uses __import__, importlib (any import of importlib or a submodule or alias, any from-import, any call through importlib, getattr(importlib, ...), or a bare alias assignment), importlib.import_module, sys.modules[...], pkgutil.resolve_name, runpy, _imp, zipimport, imp, exec, eval, or compile, or
uses ftplib / smtplib. Any use of sys.meta_path, sys.path_hooks, or sys.path_importer_cache is True, as are vars(sys), sys.__dict__, and getattr(sys, <anything>) (including a non-literal second argument such as 'mod' + 'ules'). Assigning sys to any other name (for example x=sys; x.meta_path.append(...)) is True, as is assigning getattr to another name (for example f=getattr; f(sys, 'modules')). The name sys anywhere except a direct attribute access is True: [sys][0], (x,)=(sys,), x=sys if ... else ..., h(sys), and f=vars; f(sys)['modules'] are True. from sys import * is True. import sys or import sys as s, by itself, remains False, as do sys.exit, sys.stdout.write, and sys.path.append.

It is NOT True for ordinary backtest helpers or English uses of "order":

- size_orders, settle_orders, order_plan, planned_orders, target_orders
- an array named `orders` that holds share/weight deltas for a simulator
- "in order to", "date order", "sort order", "higher-order", "higher_order"

This is a precise check, not a weaker one. A real place_order, ibkr,
alpaca, tastytrade, or similar SDK/API call still fails the gate.

A SyntaxError / parse failure fails the check outright. There is no
regex-only fallback (a failed parse must not be treated as clean).

Host allowlist: every string constant is checked, including the constant
parts of f-strings and concatenated / joined string literals. There is
no exemption based on the variable name `_CONFIG_YAML` or `CONFIG_YAML`.
Protocol-relative "//..." fragments are treated as URLs only when the
host looks like a DNS name or IP, so base64 / gzip runs of "//" are
not hosts. Every URL with a scheme (http/https/ftp), including one
inside `_CONFIG_YAML` / `CONFIG_YAML` or a YAML key, is still checked
against the allowlist. A URL built
with + , an f-string, or str.join is reconstructed when every part is a
constant; if the host still cannot be determined statically (URL from a
variable, input()/env/argv, or + / f-string / join with a non-constant
host part), any network call using it is True, and a URL-scheme prefix
with no static host is True. URL-accepting readers (pandas read_csv /
read_json / read_html / read_parquet / read_excel / read_table /
read_xml and similar, numpy loadtxt / genfromtxt, urllib.request.urlretrieve)
are treated as network calls: the host is traced with the same rules,
and if it cannot be traced to a constant the result is True. BytesIO /
StringIO, a constant local path, and a file:// URL are not network URLs.
If a reader or network-call argument carries a cloud/fsspec URI scheme
other than file:// or http/https/ftp (s3, gs, gcs, az, abfs, abfss,
adl, hdfs, oss, webhdfs, and similar), the result is True regardless
of whether the host looks like a domain. A storage_options argument is
also True. A pandas/numpy reader call that unpacks **kwargs is True
unless the unpacked object is allowlisted: a local literal dict with
all-constant keys and no storage_options, found by walking from the
reader call node to its enclosing function (or the module) — never by
looking up a function name. In that scope the name may appear exactly
twice: once at its single creation as that literal, and once as **name
in the checked reader call. Any other occurrence (Load, Store, or any
binding: for/with/import/except/match/walrus/global/nonlocal and
similar), or any globals()/locals()/vars() anywhere in the code, means
True. The same is True for any use anywhere in the file of
sys.modules, import __main__, f_globals, f_locals, f_back,
sys._getframe, inspect.currentframe, inspect.stack, gc.get_objects,
gc.get_referrers, or any .__dict__ access. dict(...) counts as a
literal only if the name dict is never bound anywhere in the file
(def, class, import as, assignment, parameter, and similar);
otherwise a **dict(...) or name = dict(...) unpacking is True.
This applies only to reader calls, not to function definitions
such as _open(f, *a, **k). file:// and scheme-less local
paths stay local. http/https/ftp are still checked against the allowlist.
numpy.load / np.load are treated as readers (so np.load('s3://...') is
True); json.load / pickle.load / yaml.load are not.

Any import or call of the cloud client or filesystem libraries boto3,
botocore, s3fs, gcsfs, adlfs, fsspec, google.cloud.storage,
azure.storage, pyarrow.fs, or smart_open is True, including
fsspec.open / fsspec.filesystem / fsspec.url_to_fs (any scheme,
including file://, because the import/call itself is sufficient).
 Function-parameter URLs (e.g. load_headlines(url),
fred_csv(endpoint)) are traced back to a constant at each call site,
including keyword arguments; if any site cannot be traced to a static
host, the result is True. The host of a URL is taken from
urllib.parse.urlsplit().hostname (so userinfo such as
https://www.eia.gov@evil.com/x is evil.com, not www.eia.gov). Each
network call records a reason describing how the host was resolved (or
that it could not be). Bibliography URLs are exempt only in docstrings,
comments, and standalone documentation string expressions.

Exact GET-only hosts (www.eia.gov, www8.cao.go.jp) are approved by
Gatekeeper one by one. They are matched exactly, never as a suffix
wildcard (not *.eia.gov, not *.go.jp). POST/PUT/DELETE/PATCH to those
hosts still fails. Do not add any future host without a GK ruling;
report each candidate host to Gatekeeper individually.

A False result only clears an LLM META false positive (e.g. size_orders).
It does not, by itself, prove the file has no broker code.

A self-defined function whose name matches an API verb (e.g. execute_order
in a backtest) still returns True. It is never auto-cleared; any
false-positive must be named explicitly in the report.

Builtins tampering is an independent file-level True condition. It is
not limited to the **kwargs reader branch. Any use of the names
builtins or __builtins__ other than an import of the module
(import builtins or import builtins as <alias>) is True, including
from builtins import .... Any from-import of __dict__ or any
dunder name from io or builtins is True. A real import of the C module _io
(import _io, import _io as <alias>, _io.__dict__, from _io import open,
from _io import __dict__, or any other from-import from _io) is treated
under the same rules as io/builtins and is True. The pack-shim alias
bound by import io as _io is unaffected. Any Name occurrence (Load included) of any
import alias of builtins, io, or a real _io import is also True — including after the
alias is re-referenced (y = q; y.dict = ..., d = q.__dict__,
g = setattr; g(q, ...), [q][0].dict = ..., k = io; k.open = ...,
getattr/__setitem__/object.__setattr__, and similar). The only
exceptions are the fixed nodes of a verified pack open shim:
_orig_open = _b.open, _b.open = _open, _io.open = _open, and
_io.StringIO / _io.BytesIO inside the _open body. In a file without
a verified shim, any such alias occurrence is True. A verified shim
requires _b and _io (and _os) to be bound by exactly
import builtins as _b, io as _io, os as _os, with binding uniqueness
for those names, a verbatim pack-shim _open, and the two open
assignments; otherwise those forms are True. The os as _os packer
form is allowed, but _os remains under the binding-uniqueness check
and calls such as _os.system(...) / _os.popen(...) stay True.

Any from-import of os (or posix/nt) process-spawn functions is True:
from os import system as q; q(...), from os import popen as p, from os
import execv, from os import spawnv as s, from os import posix_spawn as
ps, from posix import system, and from os import * (which binds those
names). from os import path, from os import environ, and import os or
import os as o or import posix by themselves remain False, as do
os.path joins and other non-spawn os attributes. import nt is True.
Assigning os.system / os.popen / os.exec* / os.spawn* /
os.posix_spawn* / os.fork / os.startfile to a name is True, including
alias calls such as f=os.system; f(...) and f=o.popen; f(...) after
import os as o. A reference to those attributes in any non-call
position is also True, including a bare os.spawnl or os.execvp,
[os.system][0], and os.system as a dict value. Direct os.system(...) /
os.popen(...) / posix.system(...) stay True as before. getattr(os,
"system") and string-literal getattr on os generally (and the same
for a posix/nt or import alias) are True, as are os.__dict__["system"]
and vars(os)["popen"] (and any os.__dict__ or vars(os) access).

from io import open, from io import open as o, and from io import *
followed by an open assignment (o = open, or an assignment to open)
are True. from io import * alone, from io import StringIO, and
import io or import io as _io (the pack-shim form) are not flagged
by this io-open-from-import rule; the existing builtins/io Name and
_io rules are unchanged.

Known limitation (not currently flagged): ctypes and deeper reflection
(for example ctypes, object.__getattribute__, pickle.loads, and
base64-decode-then-exec)
can hide a broker or network client, or hide a write of storage_options
into a **-unpacked dict, from this static check. A False result
therefore does not prove the file is clean. This checker only clears
META false positives; it is not a safety proof. getattr(sys, <anything>),
including getattr(sys, 'mod' + 'ules'), is flagged. getattr(os,
<string literal>) and getattr(os, <anything>) are flagged. Accessing
the same objects through a non-literal construction of the module
object itself (for example getattr(__import__('sys'), 'modules') or
getattr(__import__('os'), 'system')) remains a known limitation.

SHA256 of this file is reported in every staging report; recompute it
whenever the script changes (sha256sum broker_check.py).
"""
from __future__ import annotations

import ast
import os
import re
import sys
from urllib.parse import urlsplit

API_FUNCS = {
    'place_order', 'submit_order', 'send_order', 'execute_order',
    'cancel_order', 'replace_order', 'modify_order', 'close_order',
    'create_order', 'submit_bracket_order', 'placeorder', 'submitorder',
    'sendorder', 'req_mkt_data', 'reqmkt_data', 'req_historical_data',
    'reqpositions', 'req_positions', 'reqopenorders', 'req_open_orders',
    'place_trade', 'submit_trade', 'send_trade', 'execute_trade',
    'market_order', 'limit_order', 'stop_order', 'bracket_order',
}

BACKTEST_HELPERS = {
    'size_orders', 'settle_orders', 'order_plan', 'planned_orders',
    'target_orders', 'orders', 'order', 'n_orders', 'n_order',
    'order_size', 'order_sizes', 'size_order', 'settle_order',
    'order_qty', 'order_quantity', 'order_quantities', 'order_value',
    'order_notional', 'order_book_imbalance',
    'higher_order', 'nth_order',
}

BROKER_MODULES = {
    'ib_insync', 'ibapi', 'ib_async', 'ibkr', 'interactivebrokers',
    'alpaca', 'alpaca_trade_api', 'alpaca_trade', 'alpaca_py',
    'etoro', 'ccxt', 'robin_stocks', 'robinhood', 'kiteconnect',
    'tws', 'etrade', 'tda', 'schwabdev', 'schwab', 'tradier',
    'oandapyV20', 'oanda', 'fxcmpy', 'mt5', 'MetaTrader5',
    'binance', 'python_binance', 'coinbase', 'kucoin', 'bybit', 'okx',
    'tastytrade', 'tastyworks', 'webull', 'saxo', 'saxobank',
    'degiro', 'tradestation', 'igmarkets', 'igindex', 'ig_markets',
    'revolut', 'trading212', 'plus500', 'freetrade',
}

BROKER_STRINGS = re.compile(
    r'ibkr|interactive\s*brokers|alpaca(\.markets)?|paper-api\.alpaca'
    r'|etrade\.com|api\.ibkr|client\.place_order|trading_client'
    r'|\betoro\b|robinhood|kite\.zerodha|schwab\.com/trader'
    r'|metaquotes|mt5\.|oanda\.com'
    r'|tastytrade|tastyworks|webull|saxobank|\bsaxo\b|degiro'
    r'|tradestation|igmarkets|igindex|ig\.com|iggroup|revolut'
    r'|trading212|plus500',
    re.I,
)

CLIENT_CTORS = {
    'IB', 'IBKR', 'EClient', 'StockAPI', 'TradingClient', 'RESTClient',
    'TradeApi', 'BrokerClient', 'AlpacaAPI', 'TradeClient',
}

SOCKET_MODULES = {
    'socket', 'websocket', 'websockets',
}

FTP_SMTP_MODULES = {
    'ftplib', 'smtplib',
}

# Cloud client / filesystem libraries that bypass pandas readers.
# Any import or call of these is True (Gatekeeper, 2026-11-27).
CLOUD_MODULES = (
    'boto3',
    'botocore',
    's3fs',
    'gcsfs',
    'adlfs',
    'fsspec',
    'google.cloud.storage',
    'azure.storage',
    'pyarrow.fs',
    'smart_open',
)

SUBPROCESS_MODULES = {
    'subprocess',
}

PROCESS_FUNCS = {
    'system', 'popen',
    'execl', 'execle', 'execlp', 'execlpe', 'execv', 'execve', 'execvp', 'execvpe',
    'spawnl', 'spawnle', 'spawnlp', 'spawnlpe', 'spawnv', 'spawnve', 'spawnvp', 'spawnvpe',
    'posix_spawn', 'posix_spawnp',
    'fork', 'startfile',
}

DYNAMIC_FUNCS = {
    '__import__', 'exec', 'eval', 'import_module', 'compile',
}

HTTP_CLIENT_ROOTS = {
    'requests', 'urllib', 'httpx', 'aiohttp', 'http',
}

NETWORK_FUNCS = {
    'urlopen', 'urlretrieve', 'urlread', 'download',
}

# URL-accepting readers. A constant local path or BytesIO/StringIO is not a
# network URL; an untraced argument or a URL host is checked like urlopen.
PANDAS_READERS = frozenset({
    'read_csv', 'read_json', 'read_html', 'read_parquet', 'read_excel',
    'read_table', 'read_xml', 'read_fwf', 'read_hdf', 'read_feather',
    'read_orc', 'read_sas', 'read_spss', 'read_stata', 'read_pickle',
    'read_gbq',
})
NUMPY_READERS = frozenset({
    'loadtxt', 'genfromtxt',
})
URL_READERS = PANDAS_READERS | NUMPY_READERS
READER_SOURCE_KW = frozenset({
    'filepath_or_buffer', 'io', 'path', 'pat', 'source', 'url',
    'fname', 'file', 'filename', 'path_or_buf', 'p',
})
LOCAL_IO_CTORS = frozenset({
    'bytesio', 'stringio', 'open', 'path',
})

# HTTP write methods. GET/HEAD are allowed for public data download.
HTTP_WRITE = {'post', 'put', 'delete', 'patch'}
HTTP_WRITE_LITERALS = {'POST', 'PUT', 'DELETE', 'PATCH'}
HTTP_METHODS = {'get', 'head', 'post', 'put', 'delete', 'patch', 'request', 'send'}

# Suffix allowlist: Yahoo/yfinance and FRED/ALFRED only.
SUFFIX_ALLOWED = (
    'yahoo.com', 'yimg.com', 'yfapi.net',
    'stlouisfed.org',
)

# Exact GET-only hosts approved by Gatekeeper, one by one.
# Matched exactly (no suffix wildcard). Never add a host without a ruling.
EXACT_GET_HOSTS = frozenset({
    'www.eia.gov',
    'www8.cao.go.jp',
})

URL_SCHEME = re.compile(r'(?:https?|ftp)://', re.I)
# Find a URL or protocol-relative URL; hostname is parsed with urlsplit
# (userinfo must not be mistaken for the host).
URL_FIND = re.compile(r'(?:(?:https?|ftp):)?//[^\s\'"<>\\]+', re.I)
# Any URI scheme written as scheme:// . file:// is a local path; everything
# else (s3/gs/gcs/az/abfs/abfss/adl/hdfs/oss/webhdfs/http/https/ftp/...) is remote.
# A Windows drive letter (C:\ or C:/) is not a URI scheme (no "//").
URI_SCHEME_FIND = re.compile(r'(?i)(?<![A-Za-z0-9+.-])([A-Za-z][A-Za-z0-9+.-]*)://')
LOCAL_URI_SCHEMES = frozenset({'file'})
# http/https/ftp/ftps keep the existing host allowlist. Other URI schemes
# (s3/gs/gcs/az/abfs/hdfs/...) are remote regardless of whether the host
# looks like a domain.
HOST_CHECKED_URI_SCHEMES = frozenset({'file', 'http', 'https', 'ftp', 'ftps'})
# Placeholder interpolated into unresolved f-string / join parts so we can
# tell whether the hostname itself depends on a non-constant piece.
UNRESOLVED_SENTINEL = 'x-unresolved-host.invalid'
SESSION_CTORS = {'session', 'client', 'asyncclient'}
DYNAMIC_URL_FUNCS = {'input', 'getenv', 'environ'}
YAML_URL = re.compile(
    r'(?m)^[ \t]*(?P<key>[A-Za-z_][A-Za-z0-9_]*)[ \t]*:[ \t]*(?P<url>https?://\S+)',
)

FIXED_DISCLAIMER = 'No orders, no broker APIs, no paid APIs, no .env/secrets, no git, no network needed. Never fabricate.'


def _name(id_node):
    if isinstance(id_node, ast.Name):
        return id_node.id
    if isinstance(id_node, ast.Attribute):
        return id_node.attr
    return None


def _owner_chain(node):
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return '.'.join(reversed(parts)).lower()


def _const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
        try:
            return node.value.decode('utf-8', 'ignore')
        except Exception:
            return None
    return None


def _string_parts(node):
    """Yield constant string pieces, including f-string and concat parts."""
    s = _const_str(node)
    if s is not None:
        yield s
        return
    if isinstance(node, ast.JoinedStr):
        for v in node.values:
            yield from _string_parts(v)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        yield from _string_parts(node.left)
        yield from _string_parts(node.right)
        return
    if isinstance(node, ast.FormattedValue):
        yield from _string_parts(node.value)


def _is_stringy(node):
    if _const_str(node) is not None:
        return True
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_stringy(node.left) or _is_stringy(node.right)
    return False


def _mark_string_nodes(node, out):
    out.add(id(node))
    if isinstance(node, ast.JoinedStr):
        for v in node.values:
            out.add(id(v))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        _mark_string_nodes(node.left, out)
        _mark_string_nodes(node.right, out)


def _bibliography_nodes(tree):
    out = set()

    def take_expr(expr):
        if isinstance(expr, ast.Expr) and _is_stringy(expr.value):
            _mark_string_nodes(expr.value, out)

    for n in [tree, *ast.walk(tree)]:
        body = getattr(n, 'body', None)
        if isinstance(body, list):
            for stmt in body:
                take_expr(stmt)
    return out


def _url_hostname(url: str):
    """Return the URL host via urlsplit().hostname (userinfo is not the host)."""
    if not url:
        return None
    text = url.strip().rstrip('.,;:)"\']>')
    if text.startswith('//'):
        text = 'http:' + text
    try:
        host = urlsplit(text).hostname
    except ValueError:
        return None
    if not host:
        return None
    return _norm_host(host)


def _looks_like_hostname(host: str) -> bool:
    """True for DNS names (must contain a dot), IPv4/IPv6, or localhost.

    Scheme-less "//token" matches appear constantly inside base64/gzip
    blobs; a single alphanumeric label is not treated as a host.
    """
    h = _norm_host(host)
    if not h:
        return False
    if h == 'localhost':
        return True
    if re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}', h):
        return True
    if ':' in h:  # IPv6 as returned by urlsplit().hostname
        return True
    if '.' not in h:
        return False
    labels = [lab for lab in h.split('.') if lab]
    if len(labels) < 2:
        return False
    return all(re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', lab) for lab in labels)


def _hosts_in(text: str):
    if not text:
        return
    seen = set()
    for m in URL_FIND.finditer(text):
        raw = m.group(0)
        host = _url_hostname(raw)
        if not host:
            continue
        # Protocol-relative "//fragment" is a URL only when the host looks
        # like a DNS name or IP. Bare "//fvjn" in base64 is not a host.
        if not URL_SCHEME.search(raw) and not _looks_like_hostname(host):
            continue
        h = _norm_host(host)
        if h not in seen:
            seen.add(h)
            yield h


def _norm_host(host: str) -> str:
    h = (host or '').lower().rstrip('.')
    return h


def _host_is_sentinel(host: str) -> bool:
    h = _norm_host(host)
    return h == UNRESOLVED_SENTINEL or UNRESOLVED_SENTINEL in h


def _allowed_host(host: str) -> bool:
    h = _norm_host(host)
    if h in EXACT_GET_HOSTS:
        return True
    if h.startswith('www.'):
        # suffix allowlist may match yahoo/fred with or without www.
        bare = h[4:]
    else:
        bare = h
    return any(bare == s or bare.endswith('.' + s) or h == s or h.endswith('.' + s)
               for s in SUFFIX_ALLOWED)


def _host_reason_phrase(hosts):
    """Describe resolved hosts; always name a non-allowlisted host as offending."""
    if not hosts:
        return 'no static host'
    parts = []
    for h in sorted(hosts):
        if _allowed_host(h):
            parts.append(h)
        else:
            parts.append(f'{h} (not allowlisted)')
    return 'host ' + ', '.join(parts)


def _static_str(node, env):
    """Fully static string, or None if any part is dynamic."""
    s = _const_str(node)
    if s is not None:
        return s
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.FormattedValue):
                return None
            p = _static_str(v, env)
            if p is None:
                return None
            parts.append(p)
        return ''.join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _static_str(node.left, env), _static_str(node.right, env)
        if a is not None and b is not None:
            return a + b
        return None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Subscript):
        key = node.slice
        key_s = _static_str(key, env) if not isinstance(key, ast.Slice) else None
        if key_s is None and isinstance(key, ast.Constant) and isinstance(key.value, str):
            key_s = key.value
        base = None
        if isinstance(node.value, ast.Name):
            base = node.value.id
        if key_s:
            if base and f'{base}.{key_s}' in env:
                return env[f'{base}.{key_s}']
            if key_s in env:
                return env[key_s]
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'join':
        sep = _static_str(node.func.value, env)
        if sep is None or not node.args:
            return None
        seq = node.args[0]
        if not isinstance(seq, (ast.List, ast.Tuple)):
            return None
        parts = []
        for elt in seq.elts:
            p = _static_str(elt, env)
            if p is None:
                return None
            parts.append(p)
        return sep.join(parts)
    return None


def _schemes_in(text: str):
    """Yield lowercased URI schemes found as scheme:// in text."""
    if not text:
        return
    for m in URI_SCHEME_FIND.finditer(text):
        yield m.group(1).lower()


def _has_remote_scheme(text: str) -> bool:
    """True when text contains a URI scheme other than file://."""
    return any(s not in LOCAL_URI_SCHEMES for s in _schemes_in(text))


def _node_has_remote_scheme(node, env) -> bool:
    """True when a reader/network argument carries a non-file URI scheme."""
    s = _static_str(node, env)
    if s and _has_remote_scheme(s):
        return True
    for p in _string_parts(node):
        if _has_remote_scheme(p):
            return True
    return False


def _call_has_storage_options(call) -> bool:
    return any((kw.arg or '').lower() == 'storage_options' for kw in (call.keywords or []))


def _remote_schemes_of_node(node, env):
    """Non-file, non-http(s)/ftp schemes on a node (s3/gs/gcs/az/abfs/hdfs/...)."""
    found = []
    seen = set()
    chunks = []
    s = _static_str(node, env)
    if s:
        chunks.append(s)
    chunks.extend(_string_parts(node))
    for text in chunks:
        for sch in _schemes_in(text):
            if sch not in HOST_CHECKED_URI_SCHEMES and sch not in seen:
                seen.add(sch)
                found.append(sch)
    return found


def _looks_urlish(node, env) -> bool:
    s = _static_str(node, env)
    if s and (URL_SCHEME.search(s) or _has_remote_scheme(s)):
        return True
    for p in _string_parts(node):
        if URL_SCHEME.search(p) or _has_remote_scheme(p) or p.lower() in ('http://', 'https://', 'ftp://', 'http:', 'https:', 'ftp:'):
            return True
    return False


def _collect_bindings(tree):
    """Map a variable name (and 'func.name') to AST value nodes assigned or looped over."""
    binds = {}

    def add(name, value, func=None):
        if not name:
            return
        if func:
            binds.setdefault(f'{func}.{name}', []).append(value)
        else:
            binds.setdefault(name, []).append(value)

    def add_comp_targets(node, func):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in node.generators:
                tgt = gen.target
                names = []
                if isinstance(tgt, ast.Name):
                    names = [tgt]
                elif isinstance(tgt, ast.Tuple):
                    names = [elt for elt in tgt.elts if isinstance(elt, ast.Name)]
                for elt in names:
                    add(elt.id, gen.iter, func)

    def walk_body(body, func=None):
        for stmt in body:
            for n in ast.walk(stmt):
                add_comp_targets(n, func)
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        add(tgt.id, stmt.value, func)
            elif isinstance(stmt, ast.For):
                tgt = stmt.target
                if isinstance(tgt, ast.Name):
                    add(tgt.id, stmt.iter, func)
                elif isinstance(tgt, ast.Tuple):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            add(elt.id, stmt.iter, func)
                walk_body(stmt.body, func)
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                add('__return__', stmt.value, func)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk_body(stmt.body, stmt.name)
            elif isinstance(stmt, ast.If):
                walk_body(stmt.body, func)
                walk_body(stmt.orelse, func)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    if isinstance(item.optional_vars, ast.Name):
                        add(item.optional_vars.id, item.context_expr, func)
                walk_body(stmt.body, func)
            elif isinstance(stmt, ast.Try):
                walk_body(stmt.body, func)
                for h in stmt.handlers:
                    walk_body(h.body, func)

    if isinstance(tree, ast.Module):
        walk_body(tree.body)
    return binds


def _const_iter_strings(node, env):
    """If node is a tuple/list of static strings, or one static string, return them."""
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for elt in node.elts:
            s = _static_str(elt, env)
            if s is None:
                return None
            out.append(s)
        return out
    s = _static_str(node, env)
    return [s] if s is not None else None


def _hosts_from_node(node, env, binds=None, func=None):
    """Return (hosts, undetermined). undetermined means a URL is being built but the host is not static."""
    s = _static_str(node, env)
    if s is not None:
        hosts = set(_norm_host(h) for h in _hosts_in(s))
        if hosts:
            return hosts, False
        # A mere scheme prefix (https://) is a fragment of a larger URL, not
        # an undetermined host by itself. A longer scheme-bearing string with
        # no parseable hostname (rare) is undetermined.
        stripped = s.strip().lower()
        if URL_SCHEME.search(s) and stripped not in (
            'http://', 'https://', 'ftp://', 'http:', 'https:', 'ftp:',
        ):
            return set(), True
        return set(), False
    binds = binds or {}
    if isinstance(node, ast.JoinedStr):
        pieces = []
        unresolved = False
        for v in node.values:
            if isinstance(v, ast.FormattedValue):
                val = v.value
                name = val.id if isinstance(val, ast.Name) else None
                resolved = None
                if name:
                    keys = ([f'{func}.{name}', name] if func else [name])
                    for k in keys:
                        for b in binds.get(k, []):
                            strs = _const_iter_strings(b, env)
                            if strs:
                                resolved = (resolved or []) + strs
                if resolved is None:
                    one = _static_str(val, env)
                    if one is not None:
                        resolved = [one]
                if resolved:
                    pieces.append(('alts', list(dict.fromkeys(resolved))))
                else:
                    pieces.append(('unk', None))
                    unresolved = True
            else:
                one = _static_str(v, env) or _const_str(v) or ''
                pieces.append(('lit', one))
        alts = ['']
        for kind, data in pieces:
            nxt = []
            if kind == 'lit':
                for a in alts:
                    nxt.append(a + (data or ''))
            elif kind == 'alts':
                for a in alts:
                    for s2 in data:
                        nxt.append(a + s2)
            else:
                for a in alts:
                    nxt.append(a + UNRESOLVED_SENTINEL)
            alts = nxt[:16]
        hosts = set()
        host_undetermined = False
        for a in alts:
            found = list(_hosts_in(a))
            real = [h for h in found if not _host_is_sentinel(h)]
            if any(_host_is_sentinel(h) for h in found) or (URL_SCHEME.search(a) and not real and not found):
                host_undetermined = True
            hosts.update(_norm_host(h) for h in real)
        if host_undetermined:
            return hosts, True
        if hosts:
            return hosts, False
        if any(kind == 'lit' and URL_SCHEME.search(data or '') for kind, data in pieces) and unresolved:
            return set(), True
        if any(URL_SCHEME.search(a) for a in alts) and not hosts:
            return set(), True
        return set(), False
    hosts = set()
    scheme = False
    for p in _string_parts(node):
        hosts.update(_norm_host(h) for h in _hosts_in(p))
        if URL_SCHEME.search(p) or p.lower() in ('http://', 'https://', 'ftp://'):
            scheme = True
    if hosts:
        return hosts, False
    if scheme or _looks_urlish(node, env):
        return set(), True
    # join / + / f-string of non-constants with no visible host
    if isinstance(node, (ast.JoinedStr, ast.BinOp)):
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.Add):
            return set(), False
        # a concat that includes a scheme-less fragment is only undetermined
        # when used as a network URL; the caller decides.
        return set(), False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'join':
        # join that we could not fully reconstruct
        parts = []
        if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            for elt in node.args[0].elts:
                parts.extend(_string_parts(elt))
        if any(URL_SCHEME.search(p) or next(_hosts_in(p), None) for p in parts):
            return set(), True
        # A join with no URL-like constant part is not itself a URL;
        # network-call sites still treat an unresolved join argument as undetermined.
        return set(), False
    return set(), False


def _build_env(tree):

    env = {}
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Assign):
            val = _static_str(node.value, env)
            if val is None:
                # still harvest YAML-like URL keys from a large constant
                raw = _const_str(node.value)
                if raw:
                    for m in YAML_URL.finditer(raw):
                        env[m.group('key')] = m.group('url').rstrip('\'"')
                continue
            for t in node.targets:
                if isinstance(t, ast.Name):
                    env[t.id] = val
                    for m in YAML_URL.finditer(val):
                        env[m.group('key')] = m.group('url').rstrip('\'"')
    # any string constant may be an embedded config / YAML document
    for node in ast.walk(tree):
        raw = _const_str(node)
        if raw and '://' in raw:
            for m in YAML_URL.finditer(raw):
                env.setdefault(m.group('key'), m.group('url').rstrip('\'"'))
    return env


def _fn_call_sites(tree):
    """Map function name -> list of Call nodes to that name in this module."""
    sites = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            sites.setdefault(node.func.id, []).append(node)
    return sites


def _is_session_ctor(node):
    if not isinstance(node, ast.Call):
        return False
    name = (_name(node.func) or '').lower()
    if name not in SESSION_CTORS:
        return False
    if isinstance(node.func, (ast.Attribute, ast.Name)):
        chain = _owner_chain(node.func)
    else:
        chain = name
    root = (chain or '').split('.')[0]
    return root in HTTP_CLIENT_ROOTS or root in {'requests', 'httpx', 'aiohttp'}


def _collect_http_client_names(tree):
    names = set(HTTP_CLIENT_ROOTS)
    def add_from(node, target):
        if _is_session_ctor(node) and isinstance(target, ast.Name):
            names.add(target.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                add_from(node.value, t)
        elif isinstance(node, ast.AnnAssign):
            add_from(node.value, node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                add_from(item.context_expr, item.optional_vars)
    return names


def _is_dynamic_source(node):
    """True if node is input(), os.environ[...], os.getenv(...), or sys.argv[...]."""
    if isinstance(node, ast.Call):
        name = (_name(node.func) or '').lower()
        chain = _owner_chain(node.func) if isinstance(node.func, (ast.Attribute, ast.Name)) else name
        if name == 'input' and (chain in ('', 'input', 'builtins.input')):
            return True
        if name == 'getenv' or chain.endswith('.getenv') or chain.endswith('environ.get'):
            return True
    if isinstance(node, ast.Subscript):
        base = node.value
        chain = _owner_chain(base) if isinstance(base, (ast.Attribute, ast.Name)) else (_name(base) or '')
        if chain.endswith('environ') or chain.endswith('argv') or chain in {'environ', 'argv'}:
            return True
        if isinstance(base, ast.Attribute) and base.attr in {'environ', 'argv'}:
            return True
    if isinstance(node, ast.Attribute):
        if node.attr in {'environ', 'argv'}:
            return True
    return False


def _brief_node(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        v = node.value
        return v if len(v) <= 80 else v[:77] + '...'
    if isinstance(node, ast.JoinedStr):
        return 'f-string'
    if isinstance(node, ast.BinOp):
        return 'concatenated string'
    if isinstance(node, ast.Subscript):
        base = _name(node.value) or _brief_node(node.value)
        key = _const_str(node.slice)
        if key is None and isinstance(node.slice, ast.Constant):
            key = str(node.slice.value)
        return f'{base}[{key!r}]' if key is not None else f'{base}[...]'
    if isinstance(node, ast.Call):
        return f'{_name(node.func) or "call"}()'
    return type(node).__name__


def _call_arg_for_param(site, params, index, name):
    if index < len(site.args):
        return site.args[index]
    for kw in site.keywords:
        if kw.arg == name:
            return kw.value
    return None



def _module_is_cloud(name: str) -> bool:
    """True if dotted name is one of the banned cloud client/filesystem modules."""
    n = (name or '').lower().strip('.')
    if not n:
        return False
    for m in CLOUD_MODULES:
        if n == m or n.startswith(m + '.'):
            return True
    return False


def _import_modules(node):
    """Yield fully-qualified module names (and ImportFrom aliases' targets)."""
    if isinstance(node, ast.Import):
        for a in node.names:
            if a.name:
                yield a.name, (a.asname or a.name.split('.')[0])
    elif isinstance(node, ast.ImportFrom):
        base = node.module or ''
        if base:
            yield base, base.split('.')[0]
        for a in node.names:
            if not a.name or a.name == '*':
                continue
            qual = f'{base}.{a.name}' if base else a.name
            yield qual, (a.asname or a.name)


def _name_is_bound_in_file(tree, name) -> bool:
    """True if `name` is bound anywhere in the file (def, class, import, assignment, param, ...)."""
    if tree is None:
        return True  # cannot prove it is unbound
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            return True
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, (ast.Store, ast.Del, ast.AugStore)):
            return True
        if isinstance(n, ast.arg) and n.arg == name:
            return True
        if isinstance(n, ast.alias):
            bound = n.asname or n.name.rsplit('.', 1)[-1]
            if bound == name:
                return True
        if isinstance(n, ast.ExceptHandler) and n.name == name:
            return True
        if isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
            return True
        if isinstance(n, ast.MatchAs) and n.name == name:
            return True
        if isinstance(n, ast.MatchStar) and n.name == name:
            return True
        if isinstance(n, ast.MatchMapping) and getattr(n, 'rest', None) == name:
            return True
    return False


def _literal_dict_keys_are_constant_and_clean(node, env, tree=None) -> bool:
    """True only for a dict / dict() literal with all-constant keys and no storage_options.

    Spreads (**x), non-constant keys, or a storage_options key are not clean.
    dict(...) is a literal only if the name dict is never bound in the file.
    Name lookup of the dict contents is intentionally not performed here.
    """
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is None:
                return False
            key = _static_str(k, env) or _const_str(k)
            if key is None or str(key).lower() == 'storage_options':
                return False
        return True
    if isinstance(node, ast.Call) and (_name(node.func) or '').lower() == 'dict':
        if _name_is_bound_in_file(tree, 'dict'):
            return False
        if node.args:
            return False
        for kw in node.keywords or []:
            if kw.arg is None:
                return False
            if kw.arg.lower() == 'storage_options':
                return False
        return True
    return False


def _parent_map(tree):
    parents = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[id(c)] = p
    return parents


def _enclosing_scope(tree, node):
    """The function that contains node, or the module. Never looked up by name."""
    parents = _parent_map(tree)
    cur = node
    while id(cur) in parents:
        p = parents[id(cur)]
        if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return p
        cur = p
    return tree


REFLECT_ATTRS = frozenset({
    'f_globals', 'f_locals', 'f_back',
    '_getframe', 'currentframe', 'stack',
    'get_objects', 'get_referrers',
    '__dict__',
})
REFLECT_CALL_NAMES = frozenset({
    'globals', 'locals', 'vars',
    '_getframe', 'currentframe', 'stack',
    'get_objects', 'get_referrers',
})
REFLECT_IMPORTS = frozenset({
    ('sys', 'modules'), ('sys', '_getframe'),
    ('inspect', 'currentframe'), ('inspect', 'stack'),
    ('gc', 'get_objects'), ('gc', 'get_referrers'),
})


def _sys_aliases(tree):
    aliases = {'sys'}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == 'sys' or a.name.startswith('sys.'):
                    aliases.add(a.asname or a.name.split('.')[0])
    return aliases


_PACK_SHIM_IMPORT = frozenset({('builtins', '_b'), ('io', '_io'), ('os', '_os')})

_PACK_SHIM_OPEN_SOURCES = (
    'def _open(f, *a, **k):\n'
    '    try:\n'
    "        if _os.path.basename(str(f)) == 'config.yaml' and not _os.path.exists(f):\n"
    '            return _io.StringIO(_CONFIG_YAML)\n'
    '    except TypeError:\n'
    '        pass\n'
    '    return _orig_open(f, *a, **k)\n',
    'def _open(f, *a, **k):\n'
    '    try:\n'
    "        if _os.path.basename(str(f)) == 'config.yaml' and not _os.path.exists(f):\n"
    "            mode = k.get('mode', a[0] if a else 'r')\n"
    "            if 'b' in str(mode):\n"
    '                return _io.BytesIO(_CONFIG_YAML)\n'
    "            enc = k.get('encoding') or 'utf-8'\n"
    '            return _io.StringIO(_CONFIG_YAML.decode(enc))\n'
    '    except TypeError:\n'
    '        pass\n'
    '    return _orig_open(f, *a, **k)\n',
)

_PACK_SHIM_OPEN_DUMPS = tuple(
    ast.dump(ast.parse(src).body[0], include_attributes=False)
    for src in _PACK_SHIM_OPEN_SOURCES
)


def _builtins_aliases(tree):
    aliases = {'builtins', '__builtins__'}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in {'builtins', '__builtins__'} or a.name.startswith('builtins.'):
                    aliases.add(a.asname or a.name.split('.')[0])
    return aliases


def _io_aliases(tree):
    aliases = {'io'}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == 'io' or a.name.startswith('io.'):
                    aliases.add(a.asname or a.name.split('.')[0])
    return aliases


def _is_shim_open_assign(node) -> bool:
    """True only for the exact pack-shim forms _b.open = _open and _io.open = _open."""
    if not isinstance(n := node, ast.Assign) or len(n.targets) != 1:
        return False
    tgt = n.targets[0]
    if not isinstance(tgt, ast.Attribute) or tgt.attr != 'open':
        return False
    if not isinstance(tgt.value, ast.Name) or tgt.value.id not in ('_b', '_io'):
        return False
    return isinstance(n.value, ast.Name) and n.value.id == '_open'


def _is_orig_open_bind(node) -> bool:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    tgt = node.targets[0]
    if not isinstance(tgt, ast.Name) or tgt.id != '_orig_open':
        return False
    val = node.value
    return (
        isinstance(val, ast.Attribute)
        and val.attr == 'open'
        and isinstance(val.value, ast.Name)
        and val.value.id == '_b'
    )


def _pack_shim_import_node(tree):
    found = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Import):
            continue
        pairs = frozenset((a.name, a.asname) for a in n.names)
        if pairs == _PACK_SHIM_IMPORT:
            found.append(n)
    return found[0] if len(found) == 1 else None


def _name_bindings(tree, name):
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store):
            hits.append(n)
        elif isinstance(n, ast.alias):
            bound = n.asname or n.name.rsplit('.', 1)[-1]
            if bound == name:
                hits.append(n)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            hits.append(n)
        elif isinstance(n, ast.arg) and n.arg == name:
            hits.append(n)
        elif isinstance(n, ast.ExceptHandler) and n.name == name:
            hits.append(n)
        elif isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
            hits.append(n)
        elif isinstance(n, ast.MatchAs) and n.name == name:
            hits.append(n)
        elif isinstance(n, ast.MatchStar) and n.name == name:
            hits.append(n)
        elif isinstance(n, ast.MatchMapping) and getattr(n, 'rest', None) == name:
            hits.append(n)
    return hits


def _verified_pack_shim(tree) -> bool:
    """True only when the module carries the verbatim pack open shim and its exact import."""
    import_node = _pack_shim_import_node(tree)
    if import_node is None:
        return False
    opens = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_open']
    if len(opens) != 1:
        return False
    if ast.dump(opens[0], include_attributes=False) not in _PACK_SHIM_OPEN_DUMPS:
        return False
    origs = [n for n in tree.body if _is_orig_open_bind(n)]
    if len(origs) != 1:
        return False
    assign_b = assign_io = 0
    shim_assigns = []
    for n in tree.body:
        if _is_shim_open_assign(n):
            shim_assigns.append(n)
            tgt = n.targets[0].value.id
            if tgt == '_b':
                assign_b += 1
            elif tgt == '_io':
                assign_io += 1
    if assign_b != 1 or assign_io != 1:
        return False
    allowed = {id(import_node), id(opens[0]), id(origs[0])}
    for a in import_node.names:
        allowed.add(id(a))
    allowed.add(id(origs[0].targets[0]))
    for n in shim_assigns:
        allowed.add(id(n))
        allowed.add(id(n.targets[0]))
    for name in ('_b', '_io', '_os', '_open', '_orig_open'):
        for n in _name_bindings(tree, name):
            if id(n) not in allowed:
                return False
    return True


def _expr_is_name(node, names) -> bool:
    return isinstance(node, ast.Name) and node.id in names


def _is_builtins_namespace_obj(node, aliases) -> bool:
    if _expr_is_name(node, aliases):
        return True
    if isinstance(node, ast.Attribute) and node.attr == '__dict__' and _expr_is_name(node.value, aliases):
        return True
    if isinstance(node, ast.Call) and node.args:
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == 'vars' and _expr_is_name(node.args[0], aliases):
            return True
    return False


def _is_builtins_attr_target(tgt, aliases) -> bool:
    if isinstance(tgt, ast.Attribute):
        cur = tgt
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        return isinstance(cur, ast.Name) and cur.id in aliases
    if isinstance(tgt, ast.Subscript):
        return _is_builtins_namespace_obj(tgt.value, aliases)
    if isinstance(tgt, (ast.Tuple, ast.List)):
        return any(_is_builtins_attr_target(elt, aliases) for elt in tgt.elts)
    return False


def _call_func_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _module_import_aliases(tree, modules):
    """Names bound by `import <module> [as alias]` (and import module.sub as alias)."""
    aliases = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.Import):
            continue
        for a in n.names:
            if a.name in modules or any(a.name.startswith(mod + '.') for mod in modules):
                aliases.add(a.asname or a.name.split('.')[0])
    return aliases


def _verified_shim_exempt_name_nodes(tree):
    """Name nodes of _b/_io that are the fixed occurrences in a verified pack shim."""
    if not _verified_pack_shim(tree):
        return set()
    exempt = set()
    for n in tree.body:
        if _is_orig_open_bind(n) and isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name):
            exempt.add(id(n.value.value))
        if _is_shim_open_assign(n):
            tgt = n.targets[0]
            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name):
                exempt.add(id(tgt.value))
    opens = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_open']
    if len(opens) == 1:
        for n in ast.walk(opens[0]):
            if (
                isinstance(n, ast.Attribute)
                and n.attr in {'StringIO', 'BytesIO'}
                and isinstance(n.value, ast.Name)
                and n.value.id == '_io'
            ):
                exempt.add(id(n.value))
    return exempt



def _is_dunder_name(name) -> bool:
    return isinstance(name, str) and name.startswith('__') and name.endswith('__') and len(name) >= 4


def _from_import_has_dunder(node) -> bool:
    """True if an ImportFrom brings in __dict__ or any dunder name."""
    for a in node.names:
        if _is_dunder_name(a.name):
            return True
    return False



_DYNAMIC_IMPORT_ROOTS = frozenset({'importlib', 'pkgutil', 'runpy', '_imp', 'zipimport', 'imp'})


def _is_dynamic_import_module(mod) -> bool:
    """True for importlib, pkgutil, runpy, _imp, zipimport, imp, and any of their submodules."""
    if not isinstance(mod, str) or not mod:
        return False
    for root in _DYNAMIC_IMPORT_ROOTS:
        if mod == root or mod.startswith(root + '.'):
            return True
    return False


def _is_importlib_module(mod) -> bool:
    return isinstance(mod, str) and (mod == 'importlib' or mod.startswith('importlib.'))


def _getattr_literal(call):
    """If call is getattr(obj, 'literal'), return (obj, literal); else (None, None)."""
    if not isinstance(call, ast.Call) or _name(call.func) != 'getattr' or len(call.args) < 2:
        return None, None
    arg1 = call.args[1]
    if isinstance(arg1, ast.Constant) and isinstance(arg1.value, str):
        return call.args[0], arg1.value
    return None, None


_SYS_IMPORT_HOOK_ATTRS = frozenset({
    'meta_path', 'path_hooks', 'path_importer_cache', 'modules', '__dict__',
})


def _is_sys_object(node, sys_aliases) -> bool:
    return isinstance(node, ast.Name) and (node.id == 'sys' or node.id in sys_aliases)


def _assigned_name_ids(target):
    """Yield Name ids bound by an assignment target (including unpacking)."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _assigned_name_ids(target.value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _assigned_name_ids(elt)


def _assignment_value_name(value):
    return value.id if isinstance(value, ast.Name) else None


def _collect_name_assignment_aliases(tree, sources):
    """Fixed-point of names bound by assignment (or walrus) to a name in sources."""
    names = set(sources)
    changed = True
    while changed:
        changed = False
        for n in ast.walk(tree):
            pairs = []
            if isinstance(n, ast.Assign):
                pairs.append((n.targets, n.value))
            elif isinstance(n, ast.AnnAssign) and n.value is not None:
                pairs.append(([n.target], n.value))
            elif isinstance(n, ast.NamedExpr):
                pairs.append(([n.target], n.value))
            for targets, value in pairs:
                src = _assignment_value_name(value)
                if src not in names:
                    continue
                for target in targets:
                    for tid in _assigned_name_ids(target):
                        if tid not in names:
                            names.add(tid)
                            changed = True
    return names


def _file_level_importlib(tree) -> bool:
    """True for importlib/pkgutil/runpy/_imp/zipimport/imp imports and uses, sys import hooks, vars(sys), sys.__dict__, getattr(sys, ...), from sys import *, and the name sys anywhere except a direct attribute access."""
    if tree is None:
        return True
    aliases = _module_import_aliases(tree, set(_DYNAMIC_IMPORT_ROOTS))
    sys_import_aliases = _module_import_aliases(tree, {'sys'})
    # A bare `import sys` binds the name `sys`; include it even though
    # _module_import_aliases already adds it when the module is imported as sys.
    # Assignment aliases only: x=sys, y=x. `import sys as s` stays an import
    # alias and is not by itself True.
    sys_base = sys_import_aliases | {'sys'}
    sys_assign_aliases = _collect_name_assignment_aliases(tree, sys_base) - sys_base
    getattr_aliases = {'getattr'}
    getattr_assign_aliases = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ''
            if mod in {'builtins', '__builtins__'} or mod.startswith('builtins.') or mod.startswith('__builtins__.'):
                for a in n.names:
                    if a.name == 'getattr':
                        bound = a.asname or a.name
                        getattr_aliases.add(bound)
                        if a.asname:
                            getattr_assign_aliases.add(a.asname)
        elif isinstance(n, ast.Assign):
            # builtins.getattr bound to a name
            if isinstance(n.value, ast.Attribute) and n.value.attr == 'getattr':
                if isinstance(n.value.value, ast.Name) and n.value.value.id in {'builtins', '__builtins__'}:
                    for target in n.targets:
                        for tid in _assigned_name_ids(target):
                            getattr_aliases.add(tid)
                            getattr_assign_aliases.add(tid)
    getattr_assign_aliases |= _collect_name_assignment_aliases(tree, getattr_aliases) - getattr_aliases
    getattr_aliases |= getattr_assign_aliases
    vars_aliases = {'vars'}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ''
            if mod in {'builtins', '__builtins__'} or mod.startswith('builtins.') or mod.startswith('__builtins__.'):
                for a in n.names:
                    if a.name == 'vars':
                        vars_aliases.add(a.asname or a.name)
        elif isinstance(n, ast.Assign):
            if isinstance(n.value, ast.Attribute) and n.value.attr == 'vars':
                if isinstance(n.value.value, ast.Name) and n.value.value.id in {'builtins', '__builtins__'}:
                    for target in n.targets:
                        for tid in _assigned_name_ids(target):
                            vars_aliases.add(tid)
    vars_aliases |= _collect_name_assignment_aliases(tree, vars_aliases)
    # Assigning sys or getattr to any other name is itself True.
    if sys_assign_aliases or getattr_assign_aliases:
        return True
    sys_aliases = sys_base | sys_assign_aliases
    # Parent map so we can tell a bare use of the name sys from sys.<attr>.
    _sys_parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    sys_hook_aliases = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ''
            if mod == 'sys' or mod.startswith('sys.'):
                for a in n.names:
                    if a.name in _SYS_IMPORT_HOOK_ATTRS or a.name.startswith('modules.'):
                        sys_hook_aliases.add(a.asname or a.name)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if _is_dynamic_import_module(a.name):
                    return True
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ''
            if _is_dynamic_import_module(mod):
                return True
            if mod == 'sys' or mod.startswith('sys.'):
                for a in n.names:
                    if a.name == '*':
                        return True
                    if a.name in _SYS_IMPORT_HOOK_ATTRS or a.name.startswith('modules.'):
                        return True
        elif isinstance(n, ast.Name) and n.id in aliases:
            return True
        elif isinstance(n, ast.Name) and n.id in _DYNAMIC_IMPORT_ROOTS:
            # Using the canonical module name, even without a visible import.
            return True
        elif isinstance(n, ast.Name) and n.id in sys_hook_aliases:
            return True
        elif isinstance(n, ast.Name) and n.id in sys_aliases and isinstance(getattr(n, 'ctx', None), ast.Load):
            # The name sys (or an import alias such as s after `import sys as s`)
            # anywhere except a direct attribute access (sys.foo / s.foo) is True.
            # import sys / import sys as s by itself does not create a Load.
            parent = _sys_parents.get(id(n))
            if not (isinstance(parent, ast.Attribute) and parent.value is n):
                return True
        elif isinstance(n, ast.Attribute) and _is_sys_object(n.value, sys_aliases):
            if n.attr in _SYS_IMPORT_HOOK_ATTRS:
                return True
        elif isinstance(n, ast.Call):
            fn = _name(n.func)
            if fn in (getattr_aliases | {'getattr'}) and n.args:
                if _is_sys_object(n.args[0], sys_aliases):
                    # Any getattr(sys, <anything>), including a non-literal name
                    # and a getattr bound to another name (f=getattr; f(sys, ...)).
                    return True
            if fn in vars_aliases and n.args:
                if _is_sys_object(n.args[0], sys_aliases):
                    return True
            obj, lit = _getattr_literal(n)
            if obj is not None:
                obj_name = obj.id if isinstance(obj, ast.Name) else None
                chain = _owner_chain(obj) if isinstance(obj, (ast.Attribute, ast.Name)) else ''
                root = (chain or obj_name or '').split('.')[0]
                if (
                    root in _DYNAMIC_IMPORT_ROOTS
                    or root in aliases
                    or obj_name in aliases
                    or obj_name in _DYNAMIC_IMPORT_ROOTS
                ):
                    return True
            if isinstance(n.func, (ast.Attribute, ast.Name)):
                chain = _owner_chain(n.func)
            else:
                chain = _name(n.func) or ''
            root = (chain or '').split('.')[0]
            if root in _DYNAMIC_IMPORT_ROOTS or root in aliases:
                return True
            # pkgutil.resolve_name reached under another binding of that name
            if fn == 'resolve_name' and (
                root in {'pkgutil', ''} or root in aliases or 'pkgutil' in (chain or '')
            ):
                return True
    return False


def _file_level_builtins_tampering(tree) -> bool:
    """Independent file-level True: builtins/__builtins__ uses and any Name of a builtins/io import alias, except verified-shim nodes."""
    if tree is None:
        return True
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                name = a.name
                if name == '_io' or name.startswith('_io.'):
                    # Real C-module import; the pack shim binds io as _io, not import _io.
                    return True
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ''
            if mod in {'builtins', '__builtins__'} or mod.startswith('builtins.') or mod.startswith('__builtins__.'):
                return True
            if (mod == 'io' or mod.startswith('io.')) and _from_import_has_dunder(n):
                return True
            if mod == '_io' or mod.startswith('_io.'):
                return True
        elif isinstance(n, ast.Name) and n.id in {'builtins', '__builtins__'}:
            return True

    aliases = _module_import_aliases(tree, {'builtins', '__builtins__', 'io', '_io'})
    if not aliases:
        return False
    exempt = _verified_shim_exempt_name_nodes(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in aliases and id(n) not in exempt:
            return True
    return False


def _assigns_to_builtins(tree) -> bool:
    """Back-compat wrapper; builtins tampering is decided at file level."""
    return _file_level_builtins_tampering(tree)


def _uses_dynamic_namespace(tree) -> bool:
    """True if the file uses globals/locals/vars or the listed reflection APIs."""
    if tree is None:
        return True
    sys_aliases = _sys_aliases(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == '__main__' or a.name.startswith('__main__.'):
                    return True
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ''
            if mod == '__main__' or mod.startswith('__main__.'):
                return True
            for a in n.names:
                imported = a.name
                if (mod, imported) in REFLECT_IMPORTS:
                    return True
        elif isinstance(n, ast.Attribute):
            if n.attr in REFLECT_ATTRS:
                return True
            if n.attr == 'modules':
                base = n.value
                if isinstance(base, ast.Name) and base.id in sys_aliases:
                    return True
                if isinstance(base, ast.Attribute) and (_name(base) or '').split('.')[0] in sys_aliases:
                    return True
        elif isinstance(n, ast.Call):
            fn = n.func
            if isinstance(fn, ast.Name) and fn.id in REFLECT_CALL_NAMES:
                return True
            if isinstance(fn, ast.Attribute) and fn.attr in REFLECT_CALL_NAMES:
                return True
    return False


def _non_name_bindings(scope, name):
    """Bindings of name that are not ast.Name (import, except, arg, match, global, ...)."""
    hits = []
    for n in ast.walk(scope):
        if isinstance(n, ast.arg) and n.arg == name:
            hits.append(n)
        elif isinstance(n, ast.alias):
            bound = n.asname or n.name.rsplit('.', 1)[-1]
            if bound == name:
                hits.append(n)
        elif isinstance(n, ast.ExceptHandler) and n.name == name:
            hits.append(n)
        elif isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
            hits.append(n)
        elif isinstance(n, ast.MatchAs) and n.name == name:
            hits.append(n)
        elif isinstance(n, ast.MatchStar) and n.name == name:
            hits.append(n)
        elif isinstance(n, ast.MatchMapping) and getattr(n, 'rest', None) == name:
            hits.append(n)
    return hits


def _is_simple_literal_creation(name_node, tree, env) -> bool:
    """True if name_node is the sole target of `name = {constant keys}` or an annotated form."""
    parents = _parent_map(tree)
    p = parents.get(id(name_node))
    if isinstance(p, ast.Assign):
        if len(p.targets) == 1 and p.targets[0] is name_node:
            return _literal_dict_keys_are_constant_and_clean(p.value, env, tree)
        return False
    if isinstance(p, ast.AnnAssign):
        if p.target is name_node and p.value is not None:
            return _literal_dict_keys_are_constant_and_clean(p.value, env, tree)
        return False
    return False


def _name_is_allowlisted_starred_dict(name, star_node, call, tree, env) -> bool:
    """True only when name appears exactly twice in the call's own scope: literal creation + **name."""
    if tree is None or _uses_dynamic_namespace(tree):
        return False
    scope = _enclosing_scope(tree, call)
    if _non_name_bindings(scope, name):
        return False
    name_nodes = [n for n in ast.walk(scope) if isinstance(n, ast.Name) and n.id == name]
    if len(name_nodes) != 2 or star_node not in name_nodes:
        return False
    created = name_nodes[0] if name_nodes[1] is star_node else name_nodes[1]
    return _is_simple_literal_creation(created, tree, env)


def _reader_starred_kwargs_ok(call, env, binds=None, func=None, tree=None) -> bool:
    """False unless every **kw on a reader call is an allowlisted unmutated literal dict."""
    for kw in call.keywords or []:
        if kw.arg is not None:
            continue
        node = kw.value
        if _literal_dict_keys_are_constant_and_clean(node, env, tree):
            continue
        if isinstance(node, ast.Name) and _name_is_allowlisted_starred_dict(
            node.id, node, call, tree, env
        ):
            continue
        return False
    return True


def _is_url_reader(fn, chain):
    """True for pandas/numpy URL-accepting readers (and aliases like pd.read_csv)."""
    if not fn:
        return False
    key = fn.lower()
    root = (chain or '').split('.')[0].lower()
    # np.load / numpy.load only — not json.load, pickle.load, yaml.load, etc.
    if key == 'load' and root in {'np', 'numpy'}:
        return True
    if key not in URL_READERS:
        return False
    if key in NUMPY_READERS:
        return root in {'', 'np', 'numpy'} or True
    return root in {'', 'pd', 'pandas'} or True


def _reader_source_args(call):
    """The filepath/URL argument of a reader, not helper kwargs such as usecols."""
    out = []
    if call.args:
        out.append(call.args[0])
    for kw in call.keywords:
        if kw.arg and kw.arg.lower() in READER_SOURCE_KW:
            out.append(kw.value)
    return out


def _subscript_key(node, env):
    if not isinstance(node, ast.Subscript):
        return None
    sl = node.slice
    if isinstance(sl, ast.Index):  # pragma: no cover - older ast
        sl = sl.value
    return _static_str(sl, env) or _const_str(sl)


def _is_local_io_ctor(node):
    if not isinstance(node, ast.Call):
        return False
    chain = _owner_chain(node.func) if isinstance(node.func, (ast.Attribute, ast.Name)) else (_name(node.func) or '')
    tail = (chain or '').split('.')[-1].lower()
    return tail in LOCAL_IO_CTORS


def _is_network_io_ctor(node):
    if not isinstance(node, ast.Call):
        return False
    fn = _name(node.func)
    chain = _owner_chain(node.func) if isinstance(node.func, (ast.Attribute, ast.Name)) else (fn or '')
    return _is_network_call(fn, chain)


def _is_pathlike(node, env, binds=None, func=None, seen=None):
    """True for Path/CACHE joins, Path.open/gzip.open, BytesIO/StringIO, or a non-URL constant path."""
    seen = seen or set()
    if id(node) in seen:
        return False
    seen.add(id(node))
    if _is_network_io_ctor(node):
        return False
    if _node_has_remote_scheme(node, env):
        return False
    if _is_local_io_ctor(node):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return True
    if isinstance(node, ast.Call):
        chain = _owner_chain(node.func) if isinstance(node.func, (ast.Attribute, ast.Name)) else (_name(node.func) or '')
        tail = (chain or '').split('.')[-1].lower()
        if tail in LOCAL_IO_CTORS or tail == 'open':
            return True
        if tail in {'items', 'values', 'keys'} and isinstance(node.func, ast.Attribute):
            return _is_pathlike(node.func.value, env, binds, func, seen)
        fn = _name(node.func)
        if fn:
            rets = (binds or {}).get(f'{fn}.__return__', [])
            if rets and all(_is_pathlike(r, env, binds, fn, seen) for r in rets):
                return True
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return _is_pathlike(node.elt, env, binds, func, seen)
    if isinstance(node, ast.DictComp):
        return _is_pathlike(node.value, env, binds, func, seen)
    if isinstance(node, ast.Subscript):
        return _is_pathlike(node.value, env, binds, func, seen)
    if isinstance(node, ast.Attribute) and node.attr in {'open', 'name', 'parent', 'resolve'}:
        return True
    if isinstance(node, ast.Name):
        keys = []
        if func:
            keys.append(f'{func}.{node.id}')
        keys.append(node.id)
        for k in keys:
            for b in (binds or {}).get(k, []):
                # Resolve the bound value in this function's context first.
                if _is_pathlike(b, env, binds, func, seen):
                    return True
        s = (env or {}).get(node.id)
        if isinstance(s, str) and not URL_SCHEME.search(s) and not _has_remote_scheme(s) and not next(_hosts_in(s), None):
            return True
    s = _static_str(node, env)
    if s is not None and not URL_SCHEME.search(s) and not _has_remote_scheme(s) and not next(_hosts_in(s), None):
        return True
    return False


def _is_local_reader_source(node, env, binds=None, func=None):
    """True when the reader argument is BytesIO/StringIO/open/Path or a non-URL constant path."""
    return _is_pathlike(node, env, binds, func)


def _reader_param_sources_are_local(node, func, env, binds, call_sites=None, func_of=None, func_params=None):
    """True when a wrapper/function parameter only receives local reader sources at call sites."""
    if not isinstance(node, ast.Name) or not func:
        return False
    params = (func_params or {}).get(func) or []
    if node.id not in params:
        # Fall back to bound values (call-site args already stored under func.param).
        values = list((binds or {}).get(f'{func}.{node.id}', []))
        if not values:
            return False
        saw_local = False
        for b in values:
            ctx = func
            if isinstance(b, ast.Name) and b.id == node.id:
                # Same name: try other functions' bindings (caller's assignment).
                ok = False
                suffix = f'.{b.id}'
                for k, vals in (binds or {}).items():
                    if not k.endswith(suffix):
                        continue
                    other = k[:-len(suffix)]
                    if not other or other == func:
                        continue
                    if any(_is_local_reader_source(v, env, binds, other) for v in vals):
                        ok = True
                        break
                if ok:
                    saw_local = True
                    continue
                continue
            if _is_dynamic_source(b):
                return False
            if _is_local_reader_source(b, env, binds, ctx):
                saw_local = True
                continue
            hosts, und = _hosts_from_node(b, env, binds, ctx)
            if hosts:
                if und or any(not _allowed_host(h) for h in hosts):
                    return False
                saw_local = True
                continue
            if und:
                return False
            return False
        return saw_local
    sites = (call_sites or {}).get(func, [])
    if not sites:
        return False
    idx = params.index(node.id)
    saw_local = False
    for site in sites:
        arg = _call_arg_for_param(site, params, idx, node.id)
        if arg is None:
            return False
        caller = func_of(site) if func_of else None
        if caller == func and isinstance(arg, ast.Name) and arg.id == node.id:
            continue
        if _is_dynamic_source(arg):
            return False
        if _is_local_reader_source(arg, env, binds, caller):
            saw_local = True
            continue
        hosts, und = _hosts_from_node(arg, env, binds, caller)
        if hosts:
            if und or any(not _allowed_host(h) for h in hosts):
                return False
            saw_local = True
            continue
        if und:
            return False
        return False
    return saw_local


def _is_network_call(fn, chain, http_clients=None):
    if not fn:
        return False
    key = fn.lower()
    root = (chain or '').split('.')[0]
    clients = http_clients or HTTP_CLIENT_ROOTS
    if key in NETWORK_FUNCS:
        # yfinance.download is the allowed Yahoo client, not an unknown host.
        if key == 'download' and root in {'yf', 'yfinance'}:
            return False
        return True
    if root in clients or root in HTTP_CLIENT_ROOTS or root in FTP_SMTP_MODULES:
        if key in HTTP_METHODS or key in NETWORK_FUNCS or key in HTTP_WRITE:
            return True
    if 'urllib' in (chain or '') or 'requests' in (chain or '') or 'httpx' in (chain or '') or 'aiohttp' in (chain or ''):
        if key in HTTP_METHODS or key in NETWORK_FUNCS:
            return True
    if root in FTP_SMTP_MODULES or key in {'ftp', 'smtp'}:
        return True
    return False


def _getattr_name(call, env):
    """If call.func is getattr(obj, name), return (obj_chain, resolved_name or None, name_undetermined)."""
    func = call.func
    if not (isinstance(func, ast.Call) and _name(func.func) == 'getattr' and len(func.args) >= 2):
        return None, None, False
    obj = func.args[0]
    chain = _owner_chain(obj) if isinstance(obj, (ast.Attribute, ast.Name)) else (_name(obj) or '')
    resolved = _static_str(func.args[1], env)
    return (chain or '').lower(), resolved, resolved is None



_OS_PROCESS_ATTRS = frozenset(PROCESS_FUNCS)
_PROCESS_MODULES = frozenset({'os', 'posix', 'nt'})


def _process_module_aliases(tree):
    return _module_import_aliases(tree, set(_PROCESS_MODULES)) | set(_PROCESS_MODULES)


def _is_process_module_obj(node, aliases) -> bool:
    return isinstance(node, ast.Name) and node.id in aliases


def _is_os_process_attr(node, os_aliases) -> bool:
    """True for os.system / o.system / posix.system and the other process-spawn attributes."""
    if isinstance(node, ast.Attribute) and node.attr in _OS_PROCESS_ATTRS:
        if _is_process_module_obj(node.value, os_aliases):
            return True
    return False


def _assignment_values(value):
    if isinstance(value, (ast.Tuple, ast.List)):
        for elt in value.elts:
            yield from _assignment_values(elt)
    elif isinstance(value, ast.Starred):
        yield from _assignment_values(value.value)
    else:
        yield value


def _const_str_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):  # pragma: no cover - py<3.8 ast
        return node.s
    return None


def _file_level_os_process(tree) -> bool:
    """True for os/posix/nt process-spawn from-imports, aliases, non-call refs, getattr, and __dict__/vars."""
    if tree is None:
        return True
    os_aliases = _process_module_aliases(tree)
    getattr_names = {'getattr'}
    vars_names = {'vars'}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ''
            if mod in {'builtins', '__builtins__'} or mod.startswith('builtins.') or mod.startswith('__builtins__.'):
                for a in n.names:
                    if a.name == 'getattr':
                        getattr_names.add(a.asname or a.name)
                    if a.name == 'vars':
                        vars_names.add(a.asname or a.name)
            root = (mod or '').split('.')[0]
            if root in _PROCESS_MODULES:
                for a in n.names:
                    base = a.name.split('.')[0]
                    if a.name == '*' or base in _OS_PROCESS_ATTRS:
                        return True
        elif isinstance(n, ast.Import):
            for a in n.names:
                root = a.name.split('.')[0]
                # import nt is True; import posix / import os alone stay False.
                if root == 'nt' or a.name.startswith('nt.'):
                    return True
    for n in ast.walk(tree):
        # Any reference to os.system / os.spawnl / os.fork / ... (call or non-call).
        if _is_os_process_attr(n, os_aliases):
            return True
        # os.__dict__ (including os.__dict__["system"])
        if isinstance(n, ast.Attribute) and n.attr == '__dict__' and _is_process_module_obj(n.value, os_aliases):
            return True
        if isinstance(n, ast.Call):
            fn = n.func
            fname = fn.id if isinstance(fn, ast.Name) else None
            # getattr(os, "system") and string-literal getattr on os generally
            if fname in getattr_names and n.args:
                if _is_process_module_obj(n.args[0], os_aliases):
                    return True
            # vars(os) / vars(os)["popen"]
            if fname in vars_names and n.args:
                if _is_process_module_obj(n.args[0], os_aliases):
                    return True
    return False


def _file_level_io_open_from_import(tree) -> bool:
    """True for from io import open [as o], and for from io import * plus an open assignment."""
    if tree is None:
        return True
    has_star = False
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ''
            if mod == 'io' or mod.startswith('io.'):
                for a in n.names:
                    if a.name == 'open' or a.name.startswith('open.'):
                        return True
                    if a.name == '*':
                        has_star = True
    if not has_star:
        return False
    for n in ast.walk(tree):
        pairs = []
        if isinstance(n, ast.Assign):
            pairs.append((n.targets, n.value))
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            pairs.append(([n.target], n.value))
        elif isinstance(n, ast.NamedExpr):
            pairs.append(([n.target], n.value))
        for targets, value in pairs:
            if any(isinstance(v, ast.Name) and v.id == 'open' for v in _assignment_values(value)):
                return True
            for target in targets:
                if isinstance(target, ast.Name) and target.id == 'open':
                    return True
                if isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name) and elt.id == 'open':
                            return True
    return False


def check_source(src: str) -> dict:
    hits = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {
            'has_broker': True,
            'evidence': f'parse failure: {e}; check fails outright (no regex-only fallback)',
            'hits': [f'parse failure: {e}'],
            'reason': f'parse failure: {e}',
        }

    if _file_level_builtins_tampering(tree):
        return {
            'has_broker': True,
            'evidence': 'builtins/__builtins__ tampering (file-level)',
            'hits': ['builtins/__builtins__ tampering (file-level)'],
            'reason': 'file-level builtins/__builtins__ use or a non-verbatim pack open shim',
        }

    if _file_level_importlib(tree):
        return {
            'has_broker': True,
            'evidence': 'importlib/pkgutil/runpy/_imp/zipimport/imp import or sys import-hook/getattr access (file-level)',
            'hits': ['importlib/pkgutil/runpy/_imp/zipimport/imp import or sys import-hook/getattr access (file-level)'],
            'reason': 'import of importlib, pkgutil, runpy, _imp, zipimport, or imp, or access to sys.modules, sys.meta_path, sys.path_hooks, sys.path_importer_cache, vars(sys), sys.__dict__, getattr(sys, ...), from sys import *, or the name sys anywhere except a direct attribute access',
        }

    if _file_level_os_process(tree):
        return {
            'has_broker': True,
            'evidence': 'os/posix/nt process-spawn (system/popen/exec/spawn/fork/startfile) from-import, alias, non-call reference, getattr, or __dict__/vars access (file-level)',
            'hits': ['os/posix/nt process-spawn from-import, alias, non-call reference, getattr, or __dict__/vars access (file-level)'],
            'reason': 'from-import of os/posix/nt system/popen/exec*/spawn*/fork/startfile (including from os import * and from posix import system), import nt, assignment or any non-call reference of those attributes, getattr(os, ...), os.__dict__, or vars(os)',
        }

    if _file_level_io_open_from_import(tree):
        return {
            'has_broker': True,
            'evidence': 'from io import open, or from io import * followed by an open assignment (file-level)',
            'hits': ['from io import open, or from io import * followed by an open assignment (file-level)'],
            'reason': 'from io import open [as ...], or from io import * plus an assignment of or to open',
        }

    docstring_ids = _bibliography_nodes(tree)
    env = _build_env(tree)
    binds = _collect_bindings(tree)
    call_sites = _fn_call_sites(tree)
    # Propagate constant / name arguments at call sites into func.param bindings
    # so f"{endpoint}?" inside fred_csv can see FRED / FRED_ARCHIVE (positional or keyword).
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params=[a.arg for a in n.args.args]
            for site in call_sites.get(n.name, []):
                for i,a in enumerate(site.args):
                    if i < len(params):
                        binds.setdefault(f'{n.name}.{params[i]}', []).append(a)
                for kw in site.keywords:
                    if kw.arg and kw.arg in params:
                        binds.setdefault(f'{n.name}.{kw.arg}', []).append(kw.value)
    defined_funcs = {}
    func_params = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_funcs.setdefault(n.name, []).append(getattr(n, 'lineno', 0))
            func_params[n.name] = [a.arg for a in n.args.args]
    http_clients = _collect_http_client_names(tree)
    reasons = []

    def resolve_url_node(node, inside_func=None, stack=None):
        """Return (hosts, undetermined, reason)."""
        stack = set() if stack is None else set(stack)
        mark = (id(node), inside_func)
        if mark in stack:
            return set(), True, f'cyclic resolution of {_brief_node(node)}'
        stack.add(mark)

        if _is_dynamic_source(node):
            return set(), True, (
                f'URL from dynamic source {_brief_node(node)} (input/env/argv); '
                f'host cannot be determined statically'
            )

        hosts, undetermined = _hosts_from_node(node, env, binds, inside_func)
        if hosts and not undetermined:
            where = f' inside {inside_func}()' if inside_func else ''
            return hosts, False, (
                f'host {", ".join(sorted(hosts))} resolved from {_brief_node(node)}{where}'
            )
        if undetermined:
            extra = f' (also saw host {", ".join(sorted(hosts))})' if hosts else ''
            where = f' inside {inside_func}()' if inside_func else ''
            return hosts, True, (
                f'host cannot be determined statically from {_brief_node(node)}{where}{extra}'
            )

        if isinstance(node, ast.Name):
            keys = [f'{inside_func}.{node.id}'] if inside_func else [node.id]
            if inside_func:
                # Module-level assignments to this name (not other functions').
                qualified = set()
                for k, vs in binds.items():
                    if k.endswith('.' + node.id) and k != node.id:
                        qualified.update(id(v) for v in vs)
                module_vals = [v for v in binds.get(node.id, []) if id(v) not in qualified]
                if module_vals:
                    # walk them under a synthetic key handled below
                    binds.setdefault(f'__module__.{node.id}', module_vals)
                    keys.append(f'__module__.{node.id}')
            all_hosts = set()
            all_und = False
            whys = []
            # Call-site arguments first, so a function that receives both an
            # allowlisted and a non-allowlisted host is not described only
            # by the first (allowlisted) site.
            if inside_func and node.id in func_params.get(inside_func, []):
                idx = func_params[inside_func].index(node.id)
                sites = call_sites.get(inside_func, [])
                param = node.id
                if not sites:
                    all_und = True
                    whys.append(
                        f'host cannot be determined statically: parameter {param} of '
                        f'{inside_func}() has no call site'
                    )
                for site in sites:
                    arg = _call_arg_for_param(site, func_params[inside_func], idx, param)
                    if arg is None:
                        all_und = True
                        whys.append(
                            f'host cannot be determined statically: parameter {param} of '
                            f'{inside_func}() is missing at a call site'
                        )
                        continue
                    h, u, why = resolve_url_node(arg, None, stack)
                    all_hosts.update(h)
                    if u:
                        all_und = True
                    if h or u:
                        whys.append(why)
            for k in keys:
                for b in binds.get(k, []):
                    if _is_dynamic_source(b):
                        all_und = True
                        whys.append(
                            f'host cannot be determined statically: {node.id} comes from {_brief_node(b)}'
                        )
                        continue
                    h, u, why = resolve_url_node(b, inside_func, stack)
                    all_hosts.update(h)
                    if u:
                        all_und = True
                    if h or u:
                        whys.append(
                            f'{_host_reason_phrase(h) if h else "host cannot be determined"} '
                            f'resolved from assignment to {k} ({why})'
                        )
                    if isinstance(b, ast.Call) and (_name(b.func) or '').lower() in {'request', 'urlopen'}:
                        if b.args:
                            h, u, why = resolve_url_node(b.args[0], inside_func, stack)
                            all_hosts.update(h)
                            if u:
                                all_und = True
                            if h or u:
                                whys.append(why)
            if node.id in env:
                hs = set(_norm_host(h) for h in _hosts_in(env[node.id]))
                if hs:
                    all_hosts.update(hs)
                    whys.append(f'{_host_reason_phrase(hs)} resolved from constant {node.id}')
                elif URL_SCHEME.search(env[node.id] or ''):
                    all_und = True
                    whys.append(
                        f'constant {node.id} has a URL scheme but no statically determined host'
                    )
            if all_und:
                extra = f' ({_host_reason_phrase(all_hosts)})' if all_hosts else ''
                return all_hosts, True, (
                    ('; '.join(dict.fromkeys(whys)) if whys else
                     f'host cannot be determined statically from {node.id}')
                    + extra
                )
            if all_hosts:
                where = ''
                if inside_func and node.id in func_params.get(inside_func, []):
                    where = f' from call-site argument to {inside_func}({node.id})'
                return all_hosts, False, _host_reason_phrase(all_hosts) + (
                    where or f' resolved from {node.id}'
                )
            return set(), False, f'{node.id} is not a resolvable URL'
        if isinstance(node, ast.Subscript):
            s = _static_str(node, env)
            if s:
                hs = set(_norm_host(h) for h in _hosts_in(s))
                if hs:
                    return hs, False, (
                        f'host {", ".join(sorted(hs))} resolved from {_brief_node(node)}'
                    )
                if URL_SCHEME.search(s):
                    return set(), True, (
                        f'{_brief_node(node)} has a URL scheme but no statically determined host'
                    )
            key = _subscript_key(node, env)
            if key and key in env and isinstance(env[key], str):
                hs = set(_norm_host(h) for h in _hosts_in(env[key]))
                if hs:
                    return hs, False, (
                        f'{_host_reason_phrase(hs)} resolved from config/YAML key {key}'
                    )
                if URL_SCHEME.search(env[key]):
                    return set(), True, (
                        f'config/YAML key {key} has a URL scheme but no statically determined host'
                    )
            if _looks_urlish(node, env):
                return set(), True, (
                    f'host cannot be determined statically from {_brief_node(node)}'
                )
            return set(), True, (
                f'host cannot be determined statically from {_brief_node(node)}'
            )
        return set(), False, f'no URL in {_brief_node(node)}'

    def enclosing_func(node, parents):
        for p in reversed(parents):
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return p.name
        return None

    # parent map for function enclosure
    parent = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[c] = n

    def func_of(n):
        cur = n
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
            cur = parent.get(cur)
        return None

    cloud_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for qual, alias in _import_modules(node):
                if _module_is_cloud(qual):
                    if alias:
                        cloud_aliases.add(alias)
                    cloud_aliases.add(qual.split('.')[0])
                    for part in qual.split('.'):
                        # keep the leftmost imported name too (import fsspec, from s3fs import ...)
                        pass
                    hits.append(f'cloud/remote filesystem import: {qual}')

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            else:
                if node.module:
                    mods = [node.module]
            for m in mods:
                root = m.split('.')[0]
                if m in BROKER_MODULES or root in BROKER_MODULES:
                    hits.append(f'broker/live SDK import: {m}')
                if m in SOCKET_MODULES or root in SOCKET_MODULES:
                    hits.append(f'socket/websocket import: {m}')
                if m in SUBPROCESS_MODULES or root in SUBPROCESS_MODULES:
                    hits.append(f'subprocess import: {m}')
                if m in FTP_SMTP_MODULES or root in FTP_SMTP_MODULES:
                    hits.append(f'outbound ftp/smtp import: {m}')

        if isinstance(node, ast.Call):
            fn = _name(node.func)
            chain = _owner_chain(node.func) if isinstance(node.func, (ast.Attribute, ast.Name)) else (fn or '')
            if _call_has_storage_options(node):
                hits.append(
                    f'cloud/remote filesystem argument storage_options in call {chain or fn or "<call>"}()'
                )
                reasons.append(
                    f'{chain or fn or "<call>"}(): storage_options is a remote filesystem argument'
                )
            g_chain, g_name, g_unk = _getattr_name(node, env)
            if g_chain is not None:
                disp = g_name or '<dynamic>'
                hits.append(f'dynamic attribute call: getattr({g_chain or "…"}, {disp!r})()')
                if (g_name or '').lower() in HTTP_WRITE or (g_name or '').upper() in HTTP_WRITE_LITERALS:
                    hits.append(f'HTTP write via getattr: {g_chain}.{g_name}()')
                fn = fn or g_name
                chain = chain or (f'{g_chain}.{g_name}' if g_name else g_chain)

            root = (chain or '').split('.')[0]
            if fn:
                key = fn.lower()
                if key not in {x.lower() for x in BACKTEST_HELPERS}:
                    if key in {x.lower() for x in API_FUNCS}:
                        extra = ''
                        if fn in defined_funcs:
                            lines = ','.join(str(x) for x in defined_funcs[fn])
                            extra = (
                                f' (self-defined {fn}() at line {lines}; '
                                f'not auto-cleared — name a per-item false positive in the report if it is only a backtest helper)'
                            )
                        hits.append(f'broker/live-order API call: {fn}(){extra}')
                    if fn in CLIENT_CTORS:
                        hits.append(f'broker/live client constructor: {fn}()')
                if key in HTTP_WRITE:
                    hits.append(f'HTTP write call: {chain or fn}()')
                if key in DYNAMIC_FUNCS:
                    if key == 'compile' and root not in ('', 'compile', 'builtins'):
                        pass  # re.compile / similar is not builtins.compile
                    else:
                        hits.append(f'dynamic import or code execution: {chain or fn}()')
                if key in PROCESS_FUNCS:
                    hits.append(f'process spawn: {chain or fn}()')
            root = (chain or '').split('.')[0]
            if root in SUBPROCESS_MODULES or 'subprocess' in (chain or ''):
                hits.append(f'subprocess call: {chain or fn}()')
            if root in FTP_SMTP_MODULES or any(k in (chain or '') for k in FTP_SMTP_MODULES):
                hits.append(f'outbound ftp/smtp call: {chain or fn}()')
            if _module_is_cloud(chain or '') or _module_is_cloud(root) or root in cloud_aliases:
                hits.append(f'cloud/remote filesystem call: {chain or fn}()')
            else:
                # fsspec.open / filesystem / url_to_fs, even if reached via an alias
                if (fn or '').lower() in {'open', 'filesystem', 'url_to_fs'} and (
                    'fsspec' in (chain or '') or root in cloud_aliases or 'fsspec' in cloud_aliases
                ):
                    hits.append(f'cloud/remote filesystem call: {chain or fn}()')
            for kw in node.keywords:
                if kw.arg == 'method':
                    lit = _static_str(kw.value, env) or _const_str(kw.value)
                    if lit and lit.upper() in HTTP_WRITE_LITERALS:
                        hits.append(f'HTTP write method={lit!r} in {chain or fn}()')
            if fn and fn.lower() == 'socket':
                hits.append(f'socket constructor: {chain}()')
            if fn and fn.lower() in {'create_connection', 'websocket', 'websockets'}:
                if 'socket' in (chain or '') or 'websocket' in (chain or ''):
                    hits.append(f'socket/websocket call: {chain}()')

            # URL-accepting readers (pandas/numpy): trace the source argument.
            is_reader = _is_url_reader(fn, chain)
            if is_reader:
                inside_r = func_of(node)
                if _call_has_storage_options(node):
                    hits.append(
                        f'cloud/remote filesystem argument storage_options in call {chain or fn}()'
                    )
                    reasons.append(
                        f'{chain or fn}(): storage_options is a remote filesystem argument'
                    )
                if not _reader_starred_kwargs_ok(node, env, binds, inside_r, tree):
                    hits.append(
                        f'unpacking **kwargs on reader call {chain or fn}() is not statically confirmed to omit storage_options'
                    )
                    reasons.append(
                        f'{chain or fn}(): **kwargs on a reader must be statically shown not to include storage_options'
                    )
                sources = _reader_source_args(node)
                if not sources:
                    hits.append(
                        f'network call {chain or fn}() with a host that cannot be determined statically'
                    )
                    reasons.append(
                        f'{chain or fn}(): reader has no filepath/URL argument that can be traced'
                    )
                for a in sources:
                    remote = _remote_schemes_of_node(a, env)
                    if remote:
                        hits.append(
                            f'non-file URI scheme in call {chain or fn}(): {", ".join(remote)}'
                        )
                        reasons.append(
                            f'{chain or fn}(): URI scheme {", ".join(remote)} is not file://'
                        )
                        continue
                    if _is_dynamic_source(a):
                        hits.append(
                            f'network call {chain or fn}() with a host that cannot be determined statically'
                        )
                        reasons.append(
                            f'{chain or fn}(): URL from dynamic source {_brief_node(a)}'
                        )
                        continue
                    is_param = (
                        isinstance(a, ast.Name)
                        and inside_r
                        and a.id in func_params.get(inside_r, [])
                    )
                    if not is_param and _is_local_reader_source(a, env, binds, inside_r):
                        continue
                    hosts, und, why = resolve_url_node(a, inside_r)
                    for host in hosts:
                        if not _allowed_host(host):
                            hits.append(
                                f'data host outside allowlist in call {chain or fn}: {host}'
                            )
                    if hosts and not und:
                        reasons.append(f'{chain or fn}(): {why}')
                        continue
                    if hosts and und:
                        # mixed / incomplete host, still report the known host
                        reasons.append(f'{chain or fn}(): {why}')
                        continue
                    # Function-parameter or wrapper: if every call-site argument is
                    # local, this is a local reader, not an untraced URL.
                    if _reader_param_sources_are_local(a, inside_r, env, binds, call_sites, func_of, func_params):
                        continue
                    hits.append(
                        f'network call {chain or fn}() with a host that cannot be determined statically'
                    )
                    reasons.append(
                        f'{chain or fn}(): {why or "reader argument has no statically determined host"}'
                    )

            # Network calls: unresolved or non-allowlisted hosts
            is_net = _is_network_call(fn, chain, http_clients)
            if g_chain is not None:
                groot = (g_chain or '').split('.')[0]
                gkey = (g_name or '').lower()
                if (
                    groot in http_clients
                    or any(k in (g_chain or '') for k in ('requests', 'urllib', 'httpx', 'aiohttp'))
                    or gkey in HTTP_METHODS or gkey in HTTP_WRITE or gkey in NETWORK_FUNCS
                    or g_unk
                ):
                    is_net = True
            inside = func_of(node)
            args = list(node.args) + [kw.value for kw in node.keywords]
            # Self-defined download()/fetch() (e.g. 197's local FRED/Yahoo helpers)
            # are network calls only if an argument is URL-like, same as bare fetch().
            if (fn or '').lower() in {'fetch', 'download'} and is_net:
                root0 = (chain or '').split('.')[0]
                if (
                    fn in defined_funcs
                    and root0 not in http_clients
                    and 'urllib' not in (chain or '')
                    and 'requests' not in (chain or '')
                    and root0 not in {'yf', 'yfinance'}
                ):
                    is_net = False
            if (fn or '').lower() in {'fetch', 'download'} and not is_net:
                for a in args:
                    h, u = _hosts_from_node(a, env, binds, inside)
                    if u or h or _looks_urlish(a, env):
                        is_net = True
                        break
            if is_net:
                # getattr(...)(url) still has args on the outer call
                saw_urlish = False
                call_reasons = []
                if _call_has_storage_options(node):
                    hits.append(
                        f'cloud/remote filesystem argument storage_options in call {chain or fn}()'
                    )
                    call_reasons.append(
                        f'{chain or fn}(): storage_options is a remote filesystem argument'
                    )
                    saw_urlish = True
                for a in args:
                    remote = _remote_schemes_of_node(a, env)
                    if remote:
                        hits.append(
                            f'non-file URI scheme in call {chain or fn}(): {", ".join(remote)}'
                        )
                        call_reasons.append(
                            f'{chain or fn}(): URI scheme {", ".join(remote)} is not file://'
                        )
                        saw_urlish = True
                    hosts, und, why = resolve_url_node(a, inside)
                    if und or hosts or _looks_urlish(a, env) or _is_dynamic_source(a):
                        saw_urlish = True
                    if und:
                        hits.append(
                            f'network call {chain or fn}() with a host that cannot be determined statically'
                        )
                        call_reasons.append(why)
                    for host in hosts:
                        if not _allowed_host(host):
                            hits.append(f'data host outside allowlist in call {chain or fn}: {host}')
                    if hosts and not und:
                        call_reasons.append(why)
                # Any network call whose host cannot be determined statically is True
                # (not only urlopen-like names).
                if not saw_urlish:
                    hits.append(
                        f'network call {chain or fn}() with a host that cannot be determined statically'
                    )
                    call_reasons.append(
                        f'network call {chain or fn}() argument has no statically determined host '
                        f'(input()/env/argv, or a variable/parameter that does not trace to a constant URL)'
                    )
                for rsn in call_reasons:
                    reasons.append(f'{chain or fn}(): {rsn}')

        # String constants / constructed URLs (assignments, f-strings, + / join)
        if isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp, ast.Call)):
            skip = False
            if isinstance(node, ast.BinOp) and not isinstance(getattr(node, 'op', None), ast.Add):
                skip = True
            if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Attribute) and node.func.attr == 'join'
            ):
                skip = True
            # Constant pieces of an f-string are resolved on the JoinedStr itself
            if isinstance(node, ast.Constant) and isinstance(parent.get(node), ast.JoinedStr):
                skip = True
            in_doc = id(node) in docstring_ids
            if not skip:
                hosts, und = _hosts_from_node(node, env, binds, func_of(node))
                parts = list(_string_parts(node)) if not isinstance(node, ast.Call) else []
                if isinstance(node, ast.Call):
                    # join
                    static = _static_str(node, env)
                    if static:
                        hosts, und = set(_norm_host(h) for h in _hosts_in(static)), False
                    else:
                        hosts, und = _hosts_from_node(node, env, binds, func_of(node))
                if not in_doc:
                    if und:
                        hits.append('URL host cannot be determined statically (concat, f-string, or join)')
                    for host in hosts:
                        if not _allowed_host(host):
                            hits.append(f'data host outside Yahoo/FRED/ALFRED allowlist: {host}')
                    for part in parts:
                        if part.strip() == FIXED_DISCLAIMER:
                            continue
                        if part in HTTP_WRITE_LITERALS:
                            hits.append(f'HTTP write literal {part!r}')
                        for m in BROKER_STRINGS.finditer(part):
                            hits.append(f'broker/live API marker {m.group(0)!r}')
                else:
                    # still honour disclaimer / skip hosts
                    pass

    seen = set()
    uniq = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        uniq.append(h)

    reason = '; '.join(dict.fromkeys(reasons))
    # If the verdict is True because of a non-allowlisted host, the reason
    # column must name that offending host — not only an allowlisted co-host.
    offenders = []
    for hit in uniq:
        m = re.search(
            r'allowlist(?: in call [^:]+)?: ([A-Za-z0-9.-]+)',
            hit,
        )
        if m and not _allowed_host(m.group(1)):
            offenders.append(m.group(1))
    missing = [h for h in dict.fromkeys(offenders) if h not in (reason or '')]
    if missing:
        named = 'offending host ' + ', '.join(missing)
        reason = (reason + '; ' if reason else '') + named
    if uniq:
        return {
            'has_broker': True,
            'evidence': '; '.join(uniq[:12]),
            'hits': uniq,
            'reason': reason or '; '.join(uniq[:12]),
        }
    return {
        'has_broker': False,
        'evidence': 'no broker/live-order API, HTTP write, socket, subprocess, dynamic import/exec, or non-allowlisted host; backtest helpers such as size_orders/settle_orders/order_plan are ignored',
        'hits': [],
        'reason': reason or 'no HTTP URL call requiring host resolution',
    }


def check_path(path: str) -> dict:
    src = open(path, errors='ignore').read()
    r = check_source(src)
    r['path'] = path
    return r


if __name__ == '__main__':
    for p in sys.argv[1:]:
        r = check_path(p)
        print(f"{r['has_broker']}\t{os.path.basename(p)}\t{r['evidence']}\t{r.get('reason', '')}")
