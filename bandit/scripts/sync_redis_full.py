#!/usr/bin/env python3
"""
Full Redis Sync Script

Fully synchronizes the `spaceplay.user_multipliers` Delta table to Azure Redis Cache.
This script is designed to be run as a Databricks Job.

Usage:
  python scripts/sync_redis_full.py \
    --table-name spaceplay.user_multipliers \
    --redis-config dbfs:/mnt/bandit/config/redis_config.json
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator

# Add parent to path for imports if needed (though this script is self-contained)
try:
    CURRENT_FILE = Path(__file__).resolve()
    BANDIT_ROOT = CURRENT_FILE.parent.parent.absolute()
except NameError:
    BANDIT_ROOT = Path("/Workspace/Users/yohan.medalsy@spaceplay.games/ai/bandit").absolute()

sys.path.insert(0, str(BANDIT_ROOT))


def _to_local_path(path: str) -> Path:
    """Convert DBFS path to local mount path."""
    if path.startswith("dbfs:/"):
        return Path(path.replace("dbfs:/", "/dbfs/"))
    return Path(path)


def load_redis_config(config_path: str) -> Dict[str, str]:
    """
    Load Redis configuration from JSON file.
    Supports Databricks secret interpolation if running in DB environment.
    """
    local_path = _to_local_path(config_path)

    # If DBFS path doesn't exist, try Workspace default location
    if not local_path.exists():
        workspace_default = Path("/Workspace/Users/yohan.medalsy@spaceplay.games/ai/bandit/config/redis_config.json")
        if workspace_default.exists():
            print(f"ℹ️  DBFS path not found, using Workspace default: {workspace_default}")
            local_path = workspace_default
        else:
            print(f"⚠️  Redis config file not found at {local_path}")
            print(f"⚠️  Also checked Workspace default: {workspace_default}")
            return {}
        
    try:
        with open(local_path, 'r') as f:
            config = json.load(f)
            
        # Handle secret interpolation if needed
        # Format: {{secrets/scope/key}}
        for k, v in config.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                secret_path = v[2:-2]
                try:
                    scope, key = secret_path.replace("secrets/", "").split("/", 1)
                    # Try to use dbutils if available
                    from pyspark.dbutils import DBUtils
                    from pyspark.sql import SparkSession
                    spark = SparkSession.builder.getOrCreate()
                    dbutils = DBUtils(spark)
                    config[k] = dbutils.secrets.get(scope=scope, key=key)
                except Exception as e:
                    print(f"   ⚠️  Failed to resolve secret {secret_path}: {e}")
                    # Keep original value if resolution fails (might be intended for local test)
                    
        return config
    except Exception as e:
        print(f"❌ Error loading Redis config: {e}")
        return {}


def process_partition(iterator: Iterator, redis_config: Dict):
    """
    Process a partition of rows and sync to Redis.
    Runs on Spark executors.
    """
    import redis
    
    host = redis_config.get("host")
    port = int(redis_config.get("port", 6380))
    key = redis_config.get("key")
    
    if not host or not key:
        print("❌ Redis configuration missing host or key")
        return

    # Connect to Redis
    try:
        r = redis.Redis(
            host=host,
            port=port,
            password=key,
            ssl=True,
            socket_timeout=10,
            retry_on_timeout=True
        )
        
        # Use pipeline for batching
        pipe = r.pipeline()
        batch_size = 1000
        count = 0
        total_synced = 0
        
        for row in iterator:
            user_id = row['user_id']
            multiplier = row['current_multiplier']
            
            # Key: "bongo:{user_id}" -> Value: multiplier
            TTL = 60 * 60 * 24 * 7  # 7 days
            pipe.set(f"bongo:{user_id}", str(multiplier), ex=TTL)

            count += 1
            
            if count >= batch_size:
                pipe.execute()
                total_synced += count
                count = 0
        
        # Execute remaining
        if count > 0:
            pipe.execute()
            total_synced += count
            
        print(f"✅ Partition synced {total_synced} records to Redis")
        
    except Exception as e:
        print(f"❌ Redis sync failed for partition: {e}")
        raise e


def main():
    parser = argparse.ArgumentParser(description="Full Redis Sync")
    parser.add_argument("--table-name", type=str, default="spaceplay.user_multipliers", help="Source Delta table")
    parser.add_argument("--redis-config", type=str, default="dbfs:/mnt/bandit/config/redis_config.json", help="Path to Redis config file")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"🚀 STARTING FULL REDIS SYNC")
    print(f"   Table: {args.table_name}")
    print(f"   Config: {args.redis_config}")
    print("=" * 80)
    
    # Load config on driver to validate
    redis_config = load_redis_config(args.redis_config)
    if not redis_config:
        print("❌ Failed to load Redis configuration. Exiting.")
        sys.exit(1)
        
    print(f"✅ Loaded Redis config (Host: {redis_config.get('host')})")
    
    # Initialize Spark
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("BanditRedisFullSync").getOrCreate()
    
    # Read table
    print(f"\n📊 Reading table {args.table_name}...")
    try:
        df = spark.table(args.table_name)
        
        # Select only needed columns
        df = df.select("user_id", "current_multiplier")
        
        row_count = df.count()
        print(f"   Found {row_count:,} rows to sync")
        
        if row_count == 0:
            print("⚠️  Table is empty. Nothing to sync.")
            sys.exit(0)
            
        # Repartition for parallelism if needed
        # Assuming ~100k rows per partition is a good balance for Redis pipelining
        num_partitions = max(1, int(row_count / 100000))
        if num_partitions > 200:
            num_partitions = 200 # Cap partitions to avoid too many small connections
            
        print(f"   Repartitioning to {num_partitions} partitions...")
        df = df.repartition(num_partitions)
        
        # Execute Sync
        print(f"\n🔄 Syncing to Redis...")
        start_time = time.time()
        
        # We need to pass the config dict to the executors
        # Spark serializes the function and its closure
        df.foreachPartition(lambda iter: process_partition(iter, redis_config))
        
        duration = time.time() - start_time
        print(f"\n✅ Full sync complete in {duration:.2f} seconds")
        print(f"   Throughput: {row_count / duration:.0f} rows/sec")
        
    except Exception as e:
        print(f"\n❌ Sync failed: {e}")
        sys.exit(1)
        
    print("\n" + "=" * 80)
    print("FULL SYNC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
