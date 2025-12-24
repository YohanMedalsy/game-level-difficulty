#!/usr/bin/env python3
"""
Phase 5: Redis Lookup Service (Simple)

Real-time API for game servers to query user difficulty multipliers from Redis.
Returns ONLY the multiplier value (float) for minimal payload.

Usage:
  python scripts/05_serve_redis_simple.py \
    --port 8000 \
    --redis-config dbfs:/mnt/bandit/config/redis_config.json
"""

import argparse
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn

app = FastAPI(
    title="DDA Redis Lookup Service (Simple)",
    description="Real-time difficulty multiplier lookups from Redis (Returns float only)",
    version="1.0.0"
)

# Global Redis client
redis_client = None


def _to_local_path(path: str) -> str:
    """Convert DBFS path to local mount path."""
    if path.startswith("dbfs:/"):
        return path.replace("dbfs:/", "/dbfs/")
    return path


def load_redis_config(config_path: str) -> dict:
    """
    Load Redis configuration from JSON file.
    Supports Databricks secret interpolation if running in DB environment.
    """
    import json
    from pathlib import Path
    
    local_path = Path(_to_local_path(config_path))
    if not local_path.exists():
        print(f"⚠️  Redis config file not found at {local_path}")
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


@app.on_event("startup")
async def connect_redis_on_startup():
    """Connect to Redis on startup."""
    global redis_client
    
    # Get Redis config from file
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-config", type=str, default="dbfs:/mnt/bandit/config/redis_config.json")
    args, _ = parser.parse_known_args()
    
    config = load_redis_config(args.redis_config)
    
    redis_host = config.get("host") or os.getenv("REDIS_HOST")
    redis_port = int(config.get("port") or os.getenv("REDIS_PORT", 6380))
    redis_key = config.get("key") or os.getenv("REDIS_KEY")
    
    if not redis_host or not redis_key:
        print("⚠️  Redis not configured. Set REDIS_HOST/KEY env vars or provide valid config file.")
        print("   Service will start but lookups will fail.")
        return
    
    try:
        import redis
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_key,
            ssl=True,
            decode_responses=True,  # Auto-decode bytes to strings
            socket_timeout=2
        )
        # Test connection
        redis_client.ping()
        print(f"✅ Connected to Redis: {redis_host}:{redis_port}")
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        redis_client = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    if redis_client:
        try:
            redis_client.ping()
            return {"status": "healthy", "redis": "connected"}
        except Exception:
            return {"status": "degraded", "redis": "disconnected"}
    return {"status": "unconfigured", "redis": "none"}


@app.get("/dda/{user_id}")
async def get_multiplier(user_id: str):
    """
    Get difficulty multiplier for a user from Redis.
    Returns just the float value.
    """
    if not redis_client:
        raise HTTPException(
            status_code=500, 
            detail="Server misconfigured: Redis not connected"
        )
    
    try:
        # Query Redis for the user's multiplier
        multiplier_key = f"bongo:{user_id}"
        multiplier_value = redis_client.get(multiplier_key)
        
        if multiplier_value is None:
            # User not found in Redis - return 404
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
        
        # Return just the float value
        # FastAPI will serialize this as a JSON number (e.g. 1.5)
        return float(multiplier_value)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid data for user {user_id}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run Redis Lookup Service (Simple)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run FastAPI on")
    parser.add_argument("--redis-config", type=str, default="dbfs:/mnt/bandit/config/redis_config.json", help="Path to Redis config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    print("=" * 80)
    print("REDIS LOOKUP SERVICE (SIMPLE)")
    print("=" * 80)
    print(f"Starting server on {args.host}:{args.port}")
    print("")
    print("Endpoints:")
    print(f"  - GET /dda/{{user_id}} - Get multiplier (float)")
    print(f"  - GET /health - Health check")
    print("=" * 80)
    
    # Patch asyncio to allow nested loops
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
