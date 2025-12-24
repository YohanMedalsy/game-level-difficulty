#!/usr/bin/env python3
"""
Full Cloudflare KV Sync Script

Fully synchronizes the `spaceplay.user_multipliers` Delta table to a Cloudflare KV Namespace
via the KV REST Bulk API.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Any

try:
    import requests
    GLOBAL_SESSION = requests.Session()
except ImportError:
    print("❌ ERROR: Python 'requests' library is required but not installed.")
    sys.exit(1)

try:
    CURRENT_FILE = Path(__file__).resolve()
    BANDIT_ROOT = CURRENT_FILE.parent.parent.absolute()
except NameError:
    BANDIT_ROOT = Path("/Workspace/Users/yohan.medalsy@spaceplay.games/ai/bandit").absolute()

sys.path.insert(0, str(BANDIT_ROOT))


def _to_local_path(path: str) -> Path:
    if path.startswith("dbfs:/"):
        return Path(path.replace("dbfs:/", "/dbfs/"))
    return Path(path)


def load_kv_config(config_path: str) -> Dict[str, str]:
    local_path = _to_local_path(config_path)
    if local_path.exists():
        try:
            with open(local_path, "r") as f:
                cfg = json.load(f)
            if "key" in cfg:
                cfg["cloudflare_api_token"] = cfg.pop("key")
            print(f"✅ Loaded KV config from local DBFS path: {local_path}")
            return cfg
        except Exception as e:
            print(f"⚠️ Failed reading local DBFS file: {e}")

    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)

        content = dbutils.fs.head(config_path)
        cfg = json.loads(content)

        if "key" in cfg:
            cfg["cloudflare_api_token"] = cfg.pop("key")

        print(f"✅ Loaded KV config from dbutils.fs.head: {config_path}")
        return cfg

    except Exception as e:
        print(f"⚠️ dbutils.fs.head failed: {e}")

    workspace_fallback = Path("/Workspace/Users/yohan.medalsy@spaceplay.games/ai/bandit/config/cloudflare_kv_config.json")

    if workspace_fallback.exists():
        try:
            with open(workspace_fallback, "r") as f:
                cfg = json.load(f)
            if "key" in cfg:
                cfg["cloudflare_api_token"] = cfg.pop("key")

            print(f"ℹ️ Using workspace fallback: {workspace_fallback}")
            return cfg
        except Exception as e:
            print(f"⚠️ Failed loading Workspace fallback config: {e}")

    print("❌ Could not load Cloudflare KV config from ANY known path.")
    return {}


def bulk_write_kv_records(kv_batch: List[Dict[str, Any]], config: Dict) -> bool:
    ACCOUNT_ID = config.get("cloudflare_account_id")
    NAMESPACE_ID = config.get("kv_namespace")
    API_TOKEN = config.get("cloudflare_api_token")

    if not all([ACCOUNT_ID, NAMESPACE_ID, API_TOKEN]):
        print("❌ Missing Cloudflare config values.")
        return False

    URL = (
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/storage/kv/"
        f"namespaces/{NAMESPACE_ID}/bulk"
    )

    HEADERS = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json'
    }

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = GLOBAL_SESSION.put(
                URL,
                headers=HEADERS,
                data=json.dumps(kv_batch),
                timeout=30
            )

            if response.ok:
                result = response.json()
                if not result.get("success"):
                    print(f"❌ KV API returned error: {result.get('errors')}")
                    return False
                return True

            if response.status_code >= 500 or response.status_code == 429:
                delay = 2 ** attempt
                print(f"⚠️ Transient KV error {response.status_code}, retrying in {delay}s")
                time.sleep(delay)
                continue

            print(f"❌ Non-retriable KV error {response.status_code}: {response.text}")
            return False

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                delay = 2 ** attempt
                print(f"⚠️ Connection error: {e}, retrying in {delay}s")
                time.sleep(delay)
                continue

            print(f"❌ Final connection error after retries: {e}")
            return False

    return False


def process_partition(iterator: Iterator, kv_config: Dict):
    TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days
    BATCH_SIZE = 10000

    kv_batch = []
    total = 0

    for row in iterator:
        kv_batch.append({
            "key": str(row["user_id"]).upper(),
            "value": str(row["current_multiplier"]),
            "expiration_ttl": TTL_SECONDS
        })

        if len(kv_batch) >= BATCH_SIZE:
            if bulk_write_kv_records(kv_batch, kv_config):
                total += len(kv_batch)
                kv_batch = []
            else:
                raise Exception("Cloudflare KV sync failed (batch).")

    if kv_batch:
        if bulk_write_kv_records(kv_batch, kv_config):
            total += len(kv_batch)
        else:
            raise Exception("Cloudflare KV sync failed (final batch).")

    print(f"✅ Synced {total} records in partition")
    yield total


def main():
    parser = argparse.ArgumentParser(description="Full Cloudflare KV Sync")
    parser.add_argument("--table-name", type=str, default="spaceplay.user_multipliers")
    parser.add_argument("--config", type=str, default="dbfs:/mnt/bandit/config/cloudflare_kv_config.json")

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 STARTING FULL CLOUDFLARE KV SYNC")
    print(f"Table:  {args.table_name}")
    print(f"Config: {args.config}")
    print("=" * 80)

    kv_config = load_kv_config(args.config)
    if not kv_config:
        print("❌ KV config failed to load.")
        sys.exit(1)

    print(f"✅ Cloudflare config loaded (Account: {kv_config.get('cloudflare_account_id')})")

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("BanditKvFullSync").getOrCreate()

    df = spark.table(args.table_name).select("user_id", "current_multiplier")

    row_count = df.count()
    print(f"📊 Rows to sync: {row_count:,}")

    if row_count == 0:
        print("⚠️ Table empty. Nothing to sync.")
        sys.exit(0)

    target_partition_size = 20000
    partitions = max(2, row_count // target_partition_size)
    partitions = min(partitions, 200)

    print(f"Repartitioning to {partitions} partitions...")
    df = df.repartition(partitions)

    print("🔄 Syncing...")
    start = time.time()

    results = df.rdd.mapPartitions(lambda it: process_partition(it, kv_config)).collect()

    duration = time.time() - start
    print(f"\n✅ Sync completed in {duration:.2f}s")
    print(f"Total processed: {sum(results):,}")
    print(f"Throughput: {row_count / duration:.0f} rows/sec")

    print("=" * 80)
    print("FULL SYNC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
