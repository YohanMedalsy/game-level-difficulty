import redis
import os

# Configuration
REDIS_HOST = "YOUR_REDIS_HOST"
REDIS_PORT = 6380
REDIS_KEY = "YOUR_REDIS_KEY"

def test_redis():
    print(f"Testing connection to {REDIS_HOST}:{REDIS_PORT}")
    print(f"Key length: {len(REDIS_KEY)}")
    print(f"Key first 5 chars: '{REDIS_KEY[:5]}'")
    print(f"Key last 5 chars: '{REDIS_KEY[-5:]}'")
    
    # Check for whitespace
    if REDIS_KEY.strip() != REDIS_KEY:
        print("⚠️  WARNING: Key has leading/trailing whitespace!")
    else:
        print("✅ Key has no leading/trailing whitespace.")

    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_KEY.strip(), # Apply fix manually here
            ssl=True,
            socket_timeout=5
        )
        print("Pinging Redis...")
        if r.ping():
            print("✅ Connection SUCCESSFUL!")
        else:
            print("❌ Ping failed (no exception raised)")
            
    except redis.AuthenticationError:
        print("❌ Authentication failed: Invalid username-password pair.")
        print("   -> The key is definitely wrong or expired.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_redis()
