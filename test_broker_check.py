#!/usr/bin/env python3
"""Regression tests for broker_check.py, including the GK 19:28 / 19:37 / 19:53 cases."""
import sys
from pathlib import Path

import broker_check as bc

FAILS = []


def expect(name, src, want, contains=None):
    r = bc.check_source(src)
    ok = r['has_broker'] is want
    if contains and contains not in r['evidence'] and contains not in ' '.join(r['hits']):
        ok = False
    if not ok:
        FAILS.append((name, want, r))
        print(f'FAIL {name}: has_broker={r["has_broker"]} want={want} evidence={r["evidence"]!r}')
    else:
        print(f'ok   {name}: has_broker={r["has_broker"]}')


# --- GK 19:28 (a): host allowlist must see string constants, not only call-arg literals
expect(
    'a_host_const_passed_to_fetch',
    'HOST_CONST="https://data.notallowed.invalid/x"\nfetch(HOST_CONST)\n',
    True,
    contains='data.notallowed.invalid',
)
expect(
    'a_fstring_constant_part',
    'x="y"\nfetch(f"https://data.notallowed.invalid/{x}")\n',
    True,
    contains='data.notallowed.invalid',
)
expect(
    'a_concat_constant_parts',
    'u="https://data.notallowed.invalid/" + "x"\nfetch(u)\n',
    True,
    contains='data.notallowed.invalid',
)
expect(
    'a_arxiv_in_docstring_only',
    '"""Research notes. See https://arxiv.org/abs/1234 and https://github.com/foo/bar."""\n'
    'def f():\n    return 1\n',
    False,
)
expect(
    'a_bare_string_expr_bibliography',
    'x = 1\n"See https://arxiv.org/abs/1234"\n',
    False,
)
expect(
    'a_midmodule_block_bibliography',
    'def f():\n    return 1\n"""Sources: https://arxiv.org/abs/1234"""\n',
    False,
)
expect(
    'a_url_in_comment_only',
    '# bibliography: https://arxiv.org/abs/1234\n'
    'def f():\n    return 1\n',
    False,
)
expect(
    'a_yahoo_url_allowed',
    'u="https://query1.finance.yahoo.com/v8/finance/chart/SPY"\nopen(u)\n',
    False,
)
expect(
    'a_fred_url_allowed',
    'u="https://api.stlouisfed.org/fred/series?id=DGS10"\nopen(u)\n',
    False,
)

# --- GK 19:28 (b): subprocess / os.system / os.popen must fail
expect(
    'b_subprocess_run',
    'import subprocess\nsubprocess.run(["echo", "1"])\n',
    True,
    contains='subprocess',
)
expect(
    'b_os_system',
    'import os\nos.system("true")\n',
    True,
    contains='system',
)
expect(
    'b_os_popen',
    'import os\nos.popen("true")\n',
    True,
    contains='popen',
)
expect(
    'b_from_subprocess_import_run',
    'from subprocess import run\nrun(["echo", "1"])\n',
    True,
    contains='subprocess',
)

# --- GK 19:28 (c): __import__, importlib.import_module, exec, eval must fail
expect(
    'c_import_concat_ib_insync',
    '__import__("ib"+"_insync")\n',
    True,
    contains='__import__',
)
expect(
    'c_importlib_import_module',
    'import importlib\nimportlib.import_module("ib_insync")\n',
    True,
    contains='importlib',
)
expect(
    'c_exec',
    'exec("print(1)")\n',
    True,
    contains='exec',
)
expect(
    'c_eval',
    'eval("1+1")\n',
    True,
    contains='eval',
)

# --- prior contracts still hold
expect('parse_failure', 'def oops(\n', True, contains='parse failure')
expect(
    'requests_post',
    'import requests\nrequests.post("https://example.com/x")\n',
    True,
)
expect(
    'urlopen_arxiv_literal',
    'from urllib.request import urlopen\nurlopen("https://arxiv.org/abs/1234")\n',
    True,
)
expect(
    'fixed_disclaimer_only',
    'x = "No orders, no broker APIs, no paid APIs, no .env/secrets, no git, no network needed. Never fabricate."\n',
    False,
)
expect(
    'size_orders_helper_only',
    'def size_orders(w): return w\nsize_orders([1, 2])\n',
    False,
)

# Self-defined execute_order is still True (never silently passed)
expect(
    'self_defined_execute_order_not_silenced',
    'def execute_order(positions, cash):\n    return positions, cash\n'
    'positions, cash = execute_order([], 1.0)\n',
    True,
    contains='execute_order',
)


expect(
    'post_column_label_not_http',
    'sessions["post"] = actual.isin(after)\nactive = sessions["pre"] | sessions["post"]\n',
    False,
)
expect(
    'uppercase_POST_assignment_still_flags',
    'METHOD = "POST"\n',
    True,
    contains='POST',
)



expect(
    'g37_re_compile_ok',
    'import re\nre.compile(r"abc")\n',
    False,
)
expect(
    'g37_yf_download_ok',
    'import yfinance as yf\nyf.download("SPY", start="2010-01-01")\n',
    False,
)


expect(
    'g37_cache_key_join_ok',
    'cache_key = "|".join(["yfinance-adjusted-close-v1", start.date().isoformat(), *symbols])\n',
    False,
)
expect(
    'g37_yahoo_host_loop',
    'import requests\nfor host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):\n    requests.get(f"https://{host}/v8/finance/chart/{t}")\n',
    False,
)
expect(
    'g37_fred_fstring_local_url',
    'from urllib.request import urlopen\nquery="id=DGS10"\nurl = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"\nurlopen(url)\n',
    False,
)
expect(
    'g37_local_fetch_helper_ok',
    'def fetch(symbol, start, end):\n    return symbol\nframe, source = fetch(symbol, start, end)\n',
    False,
)

# --- GK 19:37 (1)–(4)
expect(
    'g37_concat_split_host',
    'u = "https://ev"+"il.com"\nfetch(u)\n',
    True,
)
expect(
    'g37_fstring_interpolated_host',
    'h = "evil"\nfetch(f"https://{h}.com")\n',
    True,
)
expect(
    'g37_join_built_url',
    'u = "".join(["https://ev", "il.com"])\nfetch(u)\n',
    True,
)
expect(
    'g37_urlopen_unresolved_var',
    'from urllib.request import urlopen\nurlopen(x)\n',
    True,
)
expect(
    'g37_getattr_requests_post',
    'import requests\ngetattr(requests, "po"+"st")("https://example.com/x")\n',
    True,
)
expect(
    'g37_compile',
    'compile("print(1)", "<s>", "exec")\n',
    True,
    contains='compile',
)
expect(
    'g37_ftplib',
    'import ftplib\nftplib.FTP("ftp.example.com")\n',
    True,
    contains='ftp',
)
expect(
    'g37_smtplib',
    'import smtplib\nsmtplib.SMTP("smtp.example.com")\n',
    True,
    contains='smtp',
)
expect(
    'g37_eia_exact_host_get',
    'HOLIDAY = "https://www.eia.gov/petroleum/weekly/includes/archive.php"\nfrom urllib.request import urlopen\nurlopen(HOLIDAY)\n',
    False,
)
expect(
    'g37_cao_exact_host_get',
    'HOLIDAY_URL = "https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv"\nimport requests\nrequests.get(HOLIDAY_URL, timeout=20)\n',
    False,
)
expect(
    'g37_eia_without_www_not_allowed',
    'import requests\nrequests.get("https://eia.gov/x")\n',
    True,
)
expect(
    'g37_eia_subdomain_not_allowed',
    'import requests\nrequests.get("https://api.eia.gov/x")\n',
    True,
)
expect(
    'g37_cao_suffix_not_allowed',
    'import requests\nrequests.get("https://cao.go.jp/x")\n',
    True,
)
expect(
    'g37_eia_post_still_fails',
    'import requests\nrequests.post("https://www.eia.gov/x")\n',
    True,
    contains='post',
)

# --- GK 19:53: undetermined host always True; userinfo must use urlsplit().hostname
expect(
    'g53_input_then_requests_get',
    'u=input()\nimport requests\nrequests.get(u)\n',
    True,
    contains='cannot be determined statically',
)
expect(
    'g53_environ_then_requests_get',
    'import os, requests\nrequests.get(os.environ["U"])\n',
    True,
    contains='cannot be determined statically',
)
expect(
    'g53_argv_then_requests_get',
    'import sys, requests\nrequests.get(sys.argv[1])\n',
    True,
    contains='cannot be determined statically',
)
expect(
    'g53_userinfo_bypasses_exact_host',
    'import requests\nrequests.get("https://www.eia.gov@evil.com/x")\n',
    True,
    contains='evil.com',
)
expect(
    'g53_userinfo_does_not_treat_eia_userinfo_as_host',
    'import requests\nrequests.get("https://www.eia.gov@evil.com/x")\n',
    True,
)
# sanity: the evidence must not claim the host is the allowlisted exact host alone
r = bc.check_source('import requests\nrequests.get("https://www.eia.gov@evil.com/x")\n')
if r['has_broker'] is not True:
    FAILS.append(('g53_userinfo_true', True, r))
    print(f'FAIL g53_userinfo_true: has_broker={r["has_broker"]} evidence={r["evidence"]!r}')
elif 'www.eia.gov' in r['evidence'] and 'evil.com' not in r['evidence']:
    FAILS.append(('g53_userinfo_not_eia', True, r))
    print(f'FAIL g53_userinfo_not_eia: evidence still treats userinfo as host: {r["evidence"]!r}')
else:
    print(f'ok   g53_userinfo_host_is_evil: has_broker={r["has_broker"]} evidence={r["evidence"]!r}')

# function-parameter URLs traced to a constant at the call site stay False
expect(
    'g53_fred_csv_param_traced_to_constant',
    'from urllib.request import urlopen\n'
    'FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"\n'
    'def fred_csv(endpoint, query):\n'
    '    url = f"{endpoint}?{query}"\n'
    '    urlopen(url)\n'
    'fred_csv(FRED, "id=DGS10")\n',
    False,
)
expect(
    'g53_load_headlines_param_traced_to_config',
    'from urllib.request import urlopen\n'
    '_CONFIG_YAML = "eia_archive_url: https://www.eia.gov/petroleum/weekly/includes/archive.php\\n"\n'
    'config = {"eia_archive_url": "https://www.eia.gov/petroleum/weekly/includes/archive.php"}\n'
    'def load_headlines(url, cache_dir):\n'
    '    with urlopen(url, timeout=30) as response:\n'
    '        return response.read()\n'
    'load_headlines(config["eia_archive_url"], "cache")\n',
    False,
)
expect(
    'g53_param_with_no_call_site_is_true',
    'from urllib.request import urlopen\n'
    'def load_headlines(url):\n'
    '    urlopen(url)\n',
    True,
    contains='cannot be determined statically',
)
expect(
    'g53_fred_csv_keyword_arg_traced',
    'from urllib.request import urlopen\n'
    'FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"\n'
    'def fred_csv(endpoint, query):\n'
    '    url = f"{endpoint}?{query}"\n'
    '    urlopen(url)\n'
    'fred_csv(endpoint=FRED, query="id=DGS10")\n',
    False,
)

# GK 20:08: mixed allowlisted + non-allowlisted hosts in one function —
# verdict True is correct, but the reason column must name the offending host.
_mixed_src = (
    'from urllib.request import urlopen\n'
    'def load(url):\n'
    '    urlopen(url)\n'
    'load("https://www.eia.gov/x")\n'
    'load("https://evil.com/x")\n'
)
expect(
    'g_mixed_allow_and_disallow_is_true',
    _mixed_src,
    True,
    contains='evil.com',
)
_r = bc.check_source(_mixed_src)
if 'evil.com' not in (_r.get('reason') or ''):
    FAILS.append(('g_mixed_reason_names_offender', True, _r))
    print(f'FAIL g_mixed_reason_names_offender: reason={_r.get("reason")!r} evidence={_r["evidence"]!r}')
else:
    print(f'ok   g_mixed_reason_names_offender: has_broker={_r["has_broker"]} reason={_r.get("reason")!r}')

expect(
    'g53_local_download_helper_ok',
    'import yfinance as yf\n'
    'def download():\n'
    '    return yf.Ticker("SPY").history()\n'
    'download()\n',
    False,
)


# Embedded single-file config (gzip/base64) must not be treated as hosts
# just because the blob contains "//" runs. A real protocol-relative
# disallowed host in a network call is still True.
_embedded_cfg = (
    '_CONFIG_YAML = """market_snapshot_gzip: !!binary |\n'
    '  r27nzwjv77oyf5um6pex+riig//fvjn//l9aqn7eep38dlreme97aaaa=\n'
    '  oesbm7glgjumtqminty+lbid//yvf//s9b6rwna+ytfpzif\n'
    '"""\n'
    'print(len(_CONFIG_YAML))\n'
)
expect(
    'g_embedded_config_base64_slashes_ok',
    _embedded_cfg,
    False,
)
expect(
    'g_protocol_relative_evil_in_network_call_is_true',
    'import requests\n'
    'requests.get("//evil.com/x")\n',
    True,
    contains='evil.com',
)
expect(
    'g_https_evil_in_plain_string_still_true',
    'U = "https://evil.com/secret"\n'
    'print(U)\n',
    True,
    contains='evil.com',
)
expect(
    'g_config_yaml_does_not_hide_network_call',
    '_CONFIG_YAML = "foo: bar //fvjn//notahost"\n'
    'import requests\n'
    'requests.get("https://evil.com/x")\n',
    True,
    contains='evil.com',
)


# GK ruling on a0aad227: no name-based _CONFIG_YAML exemption.
# Scheme-bearing URLs in _CONFIG_YAML/CONFIG_YAML are still checked.
# URL-accepting readers are network calls (trace the host; untraced => True).
expect(
    'g_p6_config_yaml_url_to_read_csv_is_true',
    'import pandas as pd\n'
    '_CONFIG_YAML = "https://evil.com/x.csv"\n'
    'pd.read_csv(_CONFIG_YAML)\n',
    True,
    contains='evil.com',
)
expect(
    'g_p5_yaml_key_safe_load_read_csv_is_true',
    'import yaml\n'
    'import pandas as pd\n'
    '_CONFIG_YAML = "url: https://evil.com/x.csv\\n"\n'
    'cfg = yaml.safe_load(_CONFIG_YAML)\n'
    'pd.read_csv(cfg["url"])\n',
    True,
    contains='evil.com',
)
expect(
    'g_p10_yaml_key_safe_load_read_csv_is_true',
    'import yaml\n'
    'import pandas as pd\n'
    'CONFIG_YAML = "path: https://evil.com/data.csv\\n"\n'
    'cfg = yaml.safe_load(CONFIG_YAML)\n'
    'pd.read_csv(cfg["path"])\n',
    True,
    contains='evil.com',
)
expect(
    'g_name_spoof_config_yaml_url_is_true',
    '_CONFIG_YAML = "https://evil.com/x.csv"\n'
    'print(_CONFIG_YAML)\n',
    True,
    contains='evil.com',
)
expect(
    'g_scheme_url_in_config_yaml_blob_is_true',
    '_CONFIG_YAML = """market_snapshot_gzip: !!binary |\\n'
    '  r27nzwjv77oyf5um6pex+riig//fvjn//l9aqn7eep38dlreme97aaaa=\\n'
    'url: https://evil.com/x.csv\\n"""\n'
    'print(len(_CONFIG_YAML))\n',
    True,
    contains='evil.com',
)
expect(
    'g_read_csv_local_path_ok',
    'import pandas as pd\n'
    'pd.read_csv("prices.csv")\n',
    False,
)
expect(
    'g_read_csv_bytesio_ok',
    'import pandas as pd\n'
    'from io import BytesIO\n'
    'pd.read_csv(BytesIO(b"a,b\\n1,2"))\n',
    False,
)
expect(
    'g_read_csv_untraced_env_is_true',
    'import pandas as pd\n'
    'import os\n'
    'pd.read_csv(os.environ["U"])\n',
    True,
    contains='cannot be determined statically',
)
expect(
    'g_np_loadtxt_untraced_argv_is_true',
    'import numpy as np\n'
    'import sys\n'
    'np.loadtxt(sys.argv[1])\n',
    True,
    contains='cannot be determined statically',
)
expect(
    'g_read_csv_pathlib_join_ok',
    'from pathlib import Path\n'
    'import pandas as pd\n'
    'CACHE = Path("data_cache")\n'
    'path = CACHE / "prices.csv"\n'
    'pd.read_csv(path)\n',
    False,
)
expect(
    'g_read_csv_with_path_open_ok',
    'from pathlib import Path\n'
    'import pandas as pd\n'
    'path = Path("data_cache") / "prices.csv"\n'
    'with path.open(encoding="utf-8") as source:\n'
    '    pd.read_csv(source, index_col="date")\n',
    False,
)
expect(
    'g_read_csv_with_stringio_ok',
    'from io import StringIO\n'
    'import pandas as pd\n'
    'with StringIO("a,b\\n1,2") as stream:\n'
    '    pd.read_csv(stream)\n',
    False,
)
expect(
    'g_read_csv_dictcomp_cache_path_ok',
    'from pathlib import Path\n'
    'import pandas as pd\n'
    'CACHE = Path("data_cache")\n'
    'def load(tickers):\n'
    '    paths = {t: CACHE / f"{t}.csv" for t in tickers}\n'
    '    return {t: pd.read_csv(path) for t, path in paths.items()}\n',
    False,
)
expect(
    'g_local_read_csv_helper_bytesio_ok',
    'from io import BytesIO\n'
    'import pandas as pd\n'
    'def read_csv(source):\n'
    '    return pd.read_csv(source, index_col=0)\n'
    'frame = read_csv(BytesIO(b"a,b\\n1,2"))\n'
    'with BytesIO(b"a,b\\n1,2") as source:\n'
    '    frame = read_csv(source)\n',
    False,
)
expect(
    'g_read_csv_for_items_cache_path_ok',
    'from pathlib import Path\n'
    'import pandas as pd\n'
    'CACHE = Path("data_cache")\n'
    'def load(tickers):\n'
    '    paths = {t: CACHE / f"{t}.csv" for t in tickers}\n'
    '    prices = {}\n'
    '    for ticker, path in paths.items():\n'
    '        prices[ticker] = pd.read_csv(path)\n'
    '    return prices\n',
    False,
)
expect(
    'g_read_cache_helper_via_cache_path_ok',
    'from pathlib import Path\n'
    'import pandas as pd\n'
    'CACHE = Path("data_cache")\n'
    'def cache_path(name):\n'
    '    return CACHE / f"{name}.csv"\n'
    'def read_cache(path):\n'
    '    return pd.read_csv(path)\n'
    'def load():\n'
    '    path = cache_path("fred")\n'
    '    if path.exists():\n'
    '        return read_cache(path)\n',
    False,
)
expect(
    'g_read_csv_helper_returning_url_is_true',
    'import pandas as pd\n'
    'def cache_path(name):\n'
    '    return "https://evil.com/" + name\n'
    'def read_cache(path):\n'
    '    return pd.read_csv(path)\n'
    'def load():\n'
    '    path = cache_path("x.csv")\n'
    '    return read_cache(path)\n',
    True,
)
expect(
    'g_read_cache_mixed_local_and_evil_url_is_true',
    'from pathlib import Path\n'
    'import pandas as pd\n'
    'CACHE = Path("data_cache")\n'
    'def read_cache(path):\n'
    '    return pd.read_csv(path)\n'
    'read_cache(CACHE / "x.csv")\n'
    'read_cache("https://evil.com/x.csv")\n',
    True,
)
expect(
    'g_read_csv_s3_bucket_is_true',
    'import pandas as pd\n'
    'pd.read_csv("s3://bucket/x.csv")\n',
    True,
)
expect(
    'g_read_csv_gs_bucket_is_true',
    'import pandas as pd\n'
    'pd.read_csv("gs://bucket/x.csv")\n',
    True,
)
expect(
    'g_read_csv_gcs_bucket_is_true',
    'import pandas as pd\n'
    'pd.read_csv("gcs://bucket/x.csv")\n',
    True,
)
expect(
    'g_read_csv_az_container_is_true',
    'import pandas as pd\n'
    'pd.read_csv("az://container/x.csv")\n',
    True,
)
expect(
    'g_read_csv_abfs_container_is_true',
    'import pandas as pd\n'
    'pd.read_csv("abfs://container/x.csv")\n',
    True,
)
expect(
    'g_read_csv_abfss_container_is_true',
    'import pandas as pd\n'
    'pd.read_csv("abfss://container/x.csv")\n',
    True,
)
expect(
    'g_read_csv_hdfs_oss_webhdfs_adl_are_true',
    'import pandas as pd\n'
    'pd.read_csv("hdfs://namenode/x.csv")\n'
    'pd.read_csv("oss://bucket/x.csv")\n'
    'pd.read_csv("webhdfs://namenode/x.csv")\n'
    'pd.read_csv("adl://account/x.csv")\n',
    True,
)
expect(
    'g_read_csv_s3_fstring_is_true',
    'import pandas as pd\n'
    'bucket = "mybucket"\n'
    'pd.read_csv(f"s3://{bucket}/x.csv")\n',
    True,
)
expect(
    'g_read_csv_storage_options_on_local_path_is_true',
    'import pandas as pd\n'
    'pd.read_csv("data.csv", storage_options={"anon": True})\n',
    True,
)
expect(
    'g_read_csv_s3_with_storage_options_is_true',
    'import pandas as pd\n'
    'pd.read_csv("s3://bucket/x.csv", storage_options={"key": "v"})\n',
    True,
)
expect(
    'g_read_csv_file_scheme_is_false',
    'import pandas as pd\n'
    'pd.read_csv("file:///tmp/x.csv")\n',
    False,
)
expect(
    'g_read_csv_schemeless_local_path_is_false',
    'import pandas as pd\n'
    'pd.read_csv("/tmp/x.csv")\n'
    'pd.read_csv("data_cache/GLD.csv")\n',
    False,
)
expect(
    'g_import_boto3_is_true',
    'import boto3\n'
    'boto3.client("s3").download_file("bucket", "x.csv", "x.csv")\n',
    True,
)
expect(
    'g_import_fsspec_is_true',
    'import fsspec\n'
    'fsspec.open("s3://b/x")\n',
    True,
)
expect(
    'g_fsspec_open_s3_is_true',
    'import fsspec\n'
    'with fsspec.open("s3://b/x") as f:\n'
    '    data = f.read()\n',
    True,
)
expect(
    'g_s3fs_open_is_true',
    'import s3fs\n'
    's3fs.S3FileSystem().open("s3://b/x")\n',
    True,
)
expect(
    'g_from_s3fs_constructor_is_true',
    'from s3fs import S3FileSystem\n'
    'S3FileSystem().open("bucket/x")\n',
    True,
)
expect(
    'g_import_gcsfs_adlfs_smart_open_botocore_are_true',
    'import gcsfs, adlfs, botocore, smart_open\n',
    True,
)
expect(
    'g_import_google_cloud_storage_is_true',
    'from google.cloud import storage\n'
    'storage.Client()\n',
    True,
)
expect(
    'g_import_azure_storage_is_true',
    'from azure.storage.blob import BlobServiceClient\n',
    True,
)
expect(
    'g_import_pyarrow_fs_is_true',
    'from pyarrow import fs\n'
    'fs.S3FileSystem()\n',
    True,
)
expect(
    'g_read_csv_starred_kw_storage_options_is_true',
    'import pandas as pd\n'
    'kw = {"storage_options": {}}\n'
    'pd.read_csv("data/x.csv", **kw)\n',
    True,
)
expect(
    'g_read_csv_starred_untraced_kw_is_true',
    'import pandas as pd, os\n'
    'kw = {"sep": os.environ.get("SEP", ",")}  # untraced values, but wait: keys are static without storage_options?\n'
    'pd.read_csv("data/x.csv", **unknown_kw)\n',
    True,
)
expect(
    'g_read_csv_starred_kw_without_storage_options_is_false',
    'import pandas as pd\n'
    'kw = {"sep": ",", "index_col": 0}\n'
    'pd.read_csv("data/x.csv", **kw)\n',
    False,
)
expect(
    'g_read_csv_starred_literal_dict_no_storage_options_is_false',
    'import pandas as pd\n'
    'pd.read_csv("data/x.csv", **{"sep": ","})\n',
    False,
)
expect(
    'g_open_shim_short_def_is_true',
    'import builtins as _b, io as _io, os as _os\n'
    'def _open(f, *a, **k):\n'
    '    return _b.open(f, *a, **k)\n'
    '_b.open = _open\n',
    True,
)
expect(
    'g_open_shim_vararg_def_is_false',
    'import builtins as _b, io as _io, os as _os\n'
    '_CONFIG_YAML = b""\n'
    '_orig_open = _b.open\n'
    'def _open(f, *a, **k):\n'
    '    try:\n'
    "        if _os.path.basename(str(f)) == 'config.yaml' and not _os.path.exists(f):\n"
    '            return _io.StringIO(_CONFIG_YAML)\n'
    '    except TypeError:\n'
    '        pass\n'
    '    return _orig_open(f, *a, **k)\n'
    '_b.open = _open\n'
    '_io.open = _open\n',
    False,
)
expect(
    'g_np_load_s3_is_true',
    'import numpy as np\n'
    'np.load("s3://bucket/x.npy")\n',
    True,
)
expect(
    'g_np_load_local_path_is_false',
    'import numpy as np\n'
    'np.load("data/x.npy")\n',
    False,
)
expect(
    'g_json_load_is_not_numpy_reader',
    'import json\n'
    'json.load(open("data.json"))\n',
    False,
)
expect(
    'g_read_csv_starred_dict_subscript_mutated_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    "opts['storage_options'] = {'anon': True}\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_setdefault_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    "opts.setdefault('storage_options', {'anon': True})\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_aliased_then_mutated_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    'o2 = opts\n'
    "o2['storage_options'] = {}\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_walrus_alias_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    '(o2 := opts)\n'
    "o2['storage_options'] = {}\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_tuple_unpack_alias_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    'a, b = opts, 1\n'
    "a['storage_options'] = {}\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_list_alias_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    'l = [opts]\n'
    "l[0]['storage_options'] = {}\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_dict_alias_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    "d = {'k': opts}\n"
    "d['k']['storage_options'] = {}\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_globals_mutate_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    "globals()['opts']['storage_options'] = {}\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_for_rebind_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    "for opts in [{'storage_options': {}}]:\n"
    "    pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_with_as_rebind_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    'class _CM:\n'
    '    def __enter__(self):\n'
    "        return {'storage_options': {}}\n"
    '    def __exit__(self, *a):\n'
    '        pass\n'
    'with _CM() as opts:\n'
    "    pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_import_rebind_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    'from mymod import opts\n'
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_sibling_method_same_name_is_true',
    'import pandas as pd\n'
    'class A:\n'
    '    def f(self):\n'
    "        opts = {'sep': ','}\n"
    "        pd.read_csv('a.csv', **opts)\n"
    'class B:\n'
    '    def f(self):\n'
    "        opts = {'sep': ','}\n"
    "        opts['storage_options'] = {}\n"
    "        pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_used_only_once_is_false',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    "pd.read_csv('a.csv', **opts)\n",
    False,
)
expect(
    'g_read_csv_starred_dict_sys_modules_is_true',
    'import sys\n'
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    "sys.modules[__name__].opts['storage_options'] = {'client_kwargs': {'proxy': 'http://evil'}}\n"
    "pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_import_main_is_true',
    'import pandas as pd\n'
    'import __main__\n'
    "opts = {'sep': ','}\n"
    "__main__.opts['storage_options'] = {}\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_currentframe_f_globals_is_true',
    'import inspect\n'
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    "inspect.currentframe().f_globals['opts']['storage_options'] = {}\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_getframe_f_locals_is_true',
    'import sys\n'
    'import pandas as pd\n'
    'def f():\n'
    "    opts = {'sep': ','}\n"
    "    sys._getframe().f_locals['opts']['storage_options'] = {}\n"
    "    pd.read_csv('a.csv', **opts)\n"
    'f()\n',
    True,
)
expect(
    'g_read_csv_starred_dict_f_back_is_true',
    'import inspect\n'
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    'def f():\n'
    "    inspect.currentframe().f_back.f_globals['opts']['storage_options'] = {}\n"
    'f()\n'
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_inspect_stack_is_true',
    'import inspect\n'
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    "inspect.stack()[0].frame.f_globals['opts']['storage_options'] = {}\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_gc_get_objects_is_true',
    'import gc\n'
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    'gc.get_objects()\n'
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_gc_get_referrers_is_true',
    'import gc\n'
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    'gc.get_referrers(opts)\n'
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_dunder_dict_access_is_true',
    'import pandas as pd\n'
    "opts = {'sep': ','}\n"
    'unused = opts.__dict__ if False else None\n'
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_shadowed_dict_def_is_true',
    'import pandas as pd\n'
    'def dict(**k):\n'
    "    out = {'storage_options': {}}\n"
    '    out.update(k)\n'
    '    return out\n'
    "opts = dict(sep=',')\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_shadowed_dict_import_as_is_true',
    'import pandas as pd\n'
    'from mymod import mk as dict\n'
    "opts = dict(sep=',')\n"
    "pd.read_csv('a.csv', **opts)\n",
    True,
)
expect(
    'g_read_csv_starred_dict_builtins_dict_shadow_is_true',
    'import builtins\n'
    'import pandas as pd\n'
    'def _evil(**k):\n'
    "    out = {'storage_options': {}}\n"
    '    out.update(k)\n'
    '    return out\n'
    'builtins.dict = _evil\n'
    "opts = dict(sep=',')\n"
    "pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL', **opts)\n",
    True,
)

expect(
    'g_builtins_b1_dict_assign_fred_csv_is_true',
    'import builtins\n'
    'import pandas as pd\n'
    'builtins.dict = lambda **k: {}\n'
    "pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL')\n",
    True,
)
expect(
    'g_builtins_b2_assignment_alias_annassign_is_true',
    'import builtins\n'
    'x = builtins\n'
    'x.len: int = 3\n',
    True,
)
expect(
    'g_builtins_b3_dunder_augassign_is_true',
    '__builtins__.print += 0\n',
    True,
)
expect(
    'g_builtins_b4_setattr_is_true',
    'import builtins\n'
    "setattr(builtins, 'dict', list)\n",
    True,
)
expect(
    'g_builtins_b7_dunder_dict_subscript_is_true',
    'import builtins\n'
    "builtins.__dict__['dict'] = list\n",
    True,
)
expect(
    'g_builtins_b8_importlib_is_true',
    'import importlib\n'
    "m = importlib.import_module('builtins')\n"
    'm.dict = list\n',
    True,
)
expect(
    'g_builtins_b9_from_import_dict_is_true',
    'from builtins import __dict__ as d\n'
    "d['open'] = None\n",
    True,
)
expect(
    'g_builtins_b10_vars_subscript_is_true',
    'import builtins\n'
    "vars(builtins)['dict'] = list\n",
    True,
)
expect(
    'g_builtins_setattr_on_import_alias_is_true',
    'import builtins as b\n'
    "setattr(b, 'dict', list)\n",
    True,
)
expect(
    'g_builtins_delattr_is_true',
    'import builtins\n'
    "delattr(builtins, 'open')\n",
    True,
)
expect(
    'g_builtins_dict_update_is_true',
    'import builtins\n'
    "builtins.__dict__.update({'dict': list})\n",
    True,
)
expect(
    'g_builtins_dict_setdefault_is_true',
    'import builtins\n'
    "builtins.__dict__.setdefault('dict', list)\n",
    True,
)
expect(
    'g_builtins_vars_alias_subscript_is_true',
    'import builtins as b\n'
    "vars(b)['dict'] = list\n",
    True,
)

expect(
    'g_builtins_r1_alias_rereference_assign_is_true',
    'import builtins as q\n'
    'y = q\n'
    'y.dict = lambda **k: {}\n',
    True,
)
expect(
    'g_builtins_r2_getattr_dunder_dict_is_true',
    'import builtins as q\n'
    "getattr(q, '__dict__')['dict'] = 1\n",
    True,
)
expect(
    'g_builtins_r3_setitem_on_dunder_dict_is_true',
    'import builtins as q\n'
    "q.__dict__.__setitem__('dict', 1)\n",
    True,
)
expect(
    'g_builtins_r4_dict_alias_then_subscript_is_true',
    'import builtins as q\n'
    'd = q.__dict__\n'
    "d['dict'] = 1\n",
    True,
)
expect(
    'g_builtins_r5_vars_alias_then_subscript_is_true',
    'import builtins as q\n'
    'd = vars(q)\n'
    "d['dict'] = 1\n",
    True,
)
expect(
    'g_builtins_r6_object_setattr_is_true',
    'import builtins as q\n'
    "object.__setattr__(q, 'dict', 1)\n",
    True,
)
expect(
    'g_builtins_r7_setattr_alias_is_true',
    'import builtins as q\n'
    'g = setattr\n'
    "g(q, 'dict', 1)\n",
    True,
)
expect(
    'g_builtins_r8_list_index_attr_assign_is_true',
    'import builtins as q\n'
    '[q][0].dict = 1\n',
    True,
)
expect(
    'g_builtins_r10_io_alias_rereference_open_is_true',
    'import io\n'
    'k = io\n'
    'k.open = print\n',
    True,
)
expect(
    'g_open_shim_os_system_is_true',
    'import builtins as _b, io as _io, os as _os\n'
    '_CONFIG_YAML = b""\n'
    '_orig_open = _b.open\n'
    'def _open(f, *a, **k):\n'
    '    try:\n'
    "        if _os.path.basename(str(f)) == 'config.yaml' and not _os.path.exists(f):\n"
    '            return _io.StringIO(_CONFIG_YAML)\n'
    '    except TypeError:\n'
    '        pass\n'
    '    return _orig_open(f, *a, **k)\n'
    '_b.open = _open\n'
    '_io.open = _open\n'
    '_os.system("true")\n',
    True,
)
expect(
    'g_from_io_import_dunder_dict_is_true',
    'from io import __dict__ as d\n'
    "d['open'] = print\n",
    True,
)
expect(
    'g_from_io_import_stringio_is_false',
    'from io import StringIO\n'
    'StringIO("ab")\n',
    False,
)
expect(
    'g_from_builtins_import_dunder_name_is_true',
    'from builtins import __name__ as n\n',
    True,
)
expect(
    'g_from_importlib_import_module_alias_is_true',
    'from importlib import import_module as im\n'
    'im("os")\n',
    True,
)
expect(
    'g_from_importlib_import_dunder_import_alias_is_true',
    'from importlib import __import__ as imp\n'
    'imp("os")\n',
    True,
)
expect(
    'g_from_importlib_import_util_is_true',
    'from importlib import util\n',
    True,
)
expect(
    'g_importlib_alias_reload_call_is_true',
    'import importlib as il\n'
    'il.reload(__import__("sys"))\n',
    True,
)
expect(
    'g_import_underscore_io_is_true',
    'import _io\n',
    True,
)
expect(
    'g_import_underscore_io_as_alias_is_true',
    'import _io as z\n',
    True,
)
expect(
    'g_underscore_io_dunder_dict_is_true',
    'import _io\n'
    '_io.__dict__\n',
    True,
)
expect(
    'g_from_underscore_io_import_open_is_true',
    'from _io import open\n',
    True,
)
expect(
    'g_from_underscore_io_import_dunder_dict_is_true',
    'from _io import __dict__\n',
    True,
)
expect(
    'g_import_importlib_alias_attr_bind_is_true',
    'import importlib as il\n'
    'f = il.import_module\n',
    True,
)
expect(
    'g_getattr_importlib_import_module_is_true',
    'import importlib\n'
    "getattr(importlib, 'import_module')\n",
    True,
)
expect(
    'g_import_importlib_bare_alias_is_true',
    'import importlib as il\n',
    True,
)
expect(
    'g_import_importlib_submodule_is_true',
    'import importlib.util\n',
    True,
)
expect(
    'g_sys_modules_subscript_is_true',
    'import sys\n'
    'sys.modules["os"]\n',
    True,
)
expect(
    'g_getattr_sys_modules_literal_is_true',
    'import sys\n'
    "getattr(sys, 'modules')\n",
    True,
)
expect(
    'g_from_sys_import_modules_is_true',
    'from sys import modules\n'
    'modules["os"]\n',
    True,
)
expect(
    'g_pkgutil_resolve_name_is_true',
    'import pkgutil\n'
    'pkgutil.resolve_name("os")\n',
    True,
)
expect(
    'g_from_pkgutil_import_resolve_name_is_true',
    'from pkgutil import resolve_name\n'
    'resolve_name("os")\n',
    True,
)
expect(
    'g_import_runpy_is_true',
    'import runpy\n',
    True,
)
expect(
    'g_import_imp_c_module_is_true',
    'import _imp\n',
    True,
)
expect(
    'g_from_imp_c_module_import_is_true',
    'from _imp import acquire_lock\n',
    True,
)
expect(
    'g_import_zipimport_is_true',
    'import zipimport\n',
    True,
)
expect(
    'g_import_zipimport_as_alias_is_true',
    'import zipimport as zi\n',
    True,
)
expect(
    'g_from_zipimport_import_is_true',
    'from zipimport import zipimporter\n',
    True,
)
expect(
    'g_import_imp_stdlib_is_true',
    'import imp\n',
    True,
)
expect(
    'g_import_imp_stdlib_as_alias_is_true',
    'import imp as im\n',
    True,
)
expect(
    'g_from_imp_stdlib_import_is_true',
    'from imp import load_source\n',
    True,
)
expect(
    'g_sys_meta_path_is_true',
    'import sys\n'
    'sys.meta_path\n',
    True,
)
expect(
    'g_sys_path_hooks_is_true',
    'import sys\n'
    'sys.path_hooks\n',
    True,
)
expect(
    'g_sys_path_importer_cache_is_true',
    'import sys\n'
    'sys.path_importer_cache\n',
    True,
)
expect(
    'g_from_sys_import_meta_path_is_true',
    'from sys import meta_path\n'
    'meta_path.append(None)\n',
    True,
)
expect(
    'g_vars_sys_is_true',
    'import sys\n'
    'vars(sys)\n',
    True,
)
expect(
    'g_sys_dunder_dict_is_true',
    'import sys\n'
    'sys.__dict__\n',
    True,
)
expect(
    'g_getattr_sys_anything_is_true',
    'import sys\n'
    "getattr(sys, 'version')\n",
    True,
)
expect(
    'g_getattr_sys_mod_plus_ules_is_true',
    'import sys\n'
    "getattr(sys, 'mod' + 'ules')\n",
    True,
)
expect(
    'g_getattr_sys_alias_anything_is_true',
    'import sys as s\n'
    "getattr(s, 'modules')\n",
    True,
)
expect(
    'g_sys_assigned_to_name_meta_path_is_true',
    'import sys\n'
    'x = sys\n'
    'x.meta_path.append(None)\n',
    True,
)
expect(
    'g_getattr_assigned_to_name_sys_modules_is_true',
    'import sys\n'
    'f = getattr\n'
    "f(sys, 'modules')\n",
    True,
)
expect(
    'g_import_sys_as_alias_alone_is_false',
    'import sys as s\n',
    False,
)
expect(
    'g_sys_direct_attr_version_is_false',
    'import sys\n'
    'sys.version\n',
    False,
)
expect(
    'g_from_sys_import_star_is_true',
    'from sys import *\n',
    True,
)
expect(
    'g_sys_exit_is_false',
    'import sys\n'
    'sys.exit\n',
    False,
)
expect(
    'g_sys_stdout_write_is_false',
    'import sys\n'
    'sys.stdout.write\n',
    False,
)
expect(
    'g_sys_path_append_is_false',
    'import sys\n'
    'sys.path.append\n',
    False,
)
expect(
    'g_sys_in_list_is_true',
    'import sys\n'
    '[sys][0]\n',
    True,
)
expect(
    'g_sys_tuple_unpack_is_true',
    'import sys\n'
    '(x,)=(sys,)\n',
    True,
)
expect(
    'g_sys_ifexp_assign_is_true',
    'import sys\n'
    'x=sys if True else None\n',
    True,
)
expect(
    'g_sys_passed_to_func_is_true',
    'import sys\n'
    'def h(m):\n'
    '    return m\n'
    'h(sys)\n',
    True,
)
expect(
    'g_vars_alias_sys_modules_is_true',
    'import sys\n'
    'f=vars\n'
    "f(sys)['modules']\n",
    True,
)
expect(
    'g_read_csv_allowlisted_fred_ok',
    'import pandas as pd\n'
    'pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv")\n',
    False,
)

# real-file smoke: 514/538/574 should stay False; 511 is False after apply_fill rename.
# Packed 553 on origin/main (sha256 31fed80f47cd46a70d5cc27fe3e161e274fae0c4092940605e1ca3d9a73007e6)
# is a documented false positive: broker_check True from coincidental substrings
# ibKr (in ...QnAibKr/PTXEKxUT...), IbKr (in ...uG6IbKr8uNLZond4...), and
# DeGiro (in ...G2gNDeGirou1gJ8b2J...) in the original cache gzip literal.
# Official 553 strategy.py remains False. This expectation does not weaken any rule.
ROOT = Path('/workspace/quant_repo_staging/Quantative-Strategy/strategies')
# Use origin/main blobs so a mid-stage worktree cannot change the contract.
import subprocess, tempfile, os
from pathlib import Path as P

def origin_text(name):
    return subprocess.check_output(
        ['git', '-C', '/workspace/quant_repo_staging/Quantative-Strategy',
         'show', f'origin/main:strategies/{name}'],
        text=True,
    )

for fname, want, note in [
    ('514_risk_limited_asset_allocation_with_a_bud.py', False, '514'),
    ('538_mirror_descent_algorithms_for_risk_budge.py', False, '538'),
    ('553_large_scale_portfolio_optimization_probl.py', True, '553'),  # documented FP; packed sha256 31fed80f47cd46a70d5cc27fe3e161e274fae0c4092940605e1ca3d9a73007e6; substrings ibKr, IbKr, DeGiro
    ('574_objective_and_subjective_entropy_measure.py', False, '574'),
    ('511_dynamic_portfolio_optimization_under_cva.py', False, '511'),
    ('186_federal_holiday_effect.py', False, '186'),
    ('179_crude_oil_news_sentiment.py', False, '179'),
    ('312_cost_access_understanding_price.py', False, '312'),
]:
    src = origin_text(fname)
    r = bc.check_source(src)
    if r['has_broker'] is not want:
        FAILS.append((note, want, r))
        print(f'FAIL {note} file: has_broker={r["has_broker"]} want={want} evidence={r["evidence"]!r}')
    else:
        print(f'ok   {note} file: has_broker={r["has_broker"]} evidence={r["evidence"][:160]} reason={r.get("reason","")[:200]}')

print()
if FAILS:
    print(f'{len(FAILS)} FAIL')
    sys.exit(1)
print('ALL TESTS PASSED')
