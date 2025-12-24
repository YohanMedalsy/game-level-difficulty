#!/usr/bin/env python3
"""
Phase 5: FastAPI Inference Service

Production REST API for real-time difficulty predictions.
Calls the Databricks Model Serving endpoint for model inference.

This script is designed to run as a Databricks Job to serve the FastAPI API.

Usage in Databricks Job:
- Task type: Python script
- Script path: dbfs:/path/to/05_run_fastapi_serving.py
- Parameters: (optional) --port 8000
"""

import os
import sys
import argparse
from pathlib import Path

# Add bandit to path (robust to Databricks where __file__ may be undefined)
try:
    CURRENT_FILE = Path(__file__).resolve()
except NameError:
    import inspect
    frame = inspect.currentframe()
    CURRENT_FILE = Path(inspect.getfile(frame)).resolve() if frame else Path.cwd()

BANDIT_ROOT = CURRENT_FILE.parent.parent.absolute()
sys.path.insert(0, str(BANDIT_ROOT))

# Set up environment
os.environ.setdefault(
    "DATABRICKS_ENDPOINT_URL",
    "https://adb-249008710733422.2.azuredatabricks.net/serving-endpoints/vw-bandit-dr-endpoint/invocations"
)

# Optional: Load token from Databricks secrets if available
try:
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils()
    try:
        token = dbutils.secrets.get(scope="databricks", key="token")
        os.environ["DATABRICKS_TOKEN"] = token
    except Exception:
        pass  # No secret available
except Exception:
    pass  # Not in Databricks context

def main():
    parser = argparse.ArgumentParser(description="Run FastAPI inference service")
    parser.add_argument("--port", type=int, default=8000, help="Port to run FastAPI on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    import uvicorn
    import asyncio
    from bandit.src.serving import api

    print("=" * 80)
    print("FASTAPI INFERENCE SERVICE - PHASE 5")
    print("=" * 80)
    print(f"\n🚀 Starting FastAPI server...")
    print(f"📡 Endpoint: http://{args.host}:{args.port}")
    print(f"📖 API docs: http://{args.host}:{args.port}/docs")
    print(f"🏥 Health check: http://{args.host}:{args.port}/health")
    print(f"\n✅ Calling Databricks Model Serving endpoint:")
    print(f"   {os.environ.get('DATABRICKS_ENDPOINT_URL')}")
    print("=" * 80 + "\n")

    # Handle Databricks environment where event loop may already be running
    try:
        loop = asyncio.get_running_loop()
        # Event loop is already running (Databricks context)
        # Use nest_asyncio or run server in background
        try:
            import nest_asyncio
            nest_asyncio.apply()
            uvicorn.run(
                api.app,
                host=args.host,
                port=args.port,
                log_level="info",
                access_log=True
            )
        except ImportError:
            # Fallback: run server in a thread
            import threading
            config = uvicorn.Config(
                api.app,
                host=args.host,
                port=args.port,
                log_level="info",
                access_log=True
            )
            server = uvicorn.Server(config)
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            print("✅ FastAPI server started in background thread")
            print("⚠️  Server will run until job is stopped")
            # Keep main thread alive
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                server.should_exit = True
    except RuntimeError:
        # No event loop running, use normal uvicorn.run()
        uvicorn.run(
            api.app,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True
        )

if __name__ == "__main__":
    main()

