import asyncio
import aiohttp
import time

async def fetch(session):
    start = time.time()
    async with session.get("http://localhost:8000/health") as r:
        await r.text()
    return (time.time() - start) * 1000

async def main():
    async with aiohttp.ClientSession() as session:
        # Warmup
        await fetch(session)
        
        # 50 concurrent requests
        start = time.time()
        times = await asyncio.gather(*[fetch(session) for _ in range(50)])
        total = (time.time() - start) * 1000
        
        print(f"Requests: {len(times)}")
        print(f"Avg:      {sum(times)/len(times):.1f}ms")
        print(f"Max:      {max(times):.1f}ms")
        print(f"Min:      {min(times):.1f}ms")
        print(f"Total wall time for 50 concurrent: {total:.1f}ms")

asyncio.run(main())