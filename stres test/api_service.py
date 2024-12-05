from datetime import datetime
import aiohttp
import asyncio
from typing import Optional, Dict


async def send_request_to_endpoint(api_url: str, payload: Dict) -> Optional[Dict]:
    try:
        connector = aiohttp.TCPConnector(ssl=False)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10), connector=connector
        ) as session:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    raise ValueError(
                        f"Failed to send request: {response.status}, {await response.text()}"
                    )
                return await response.json()

    except aiohttp.ClientError as e:
        raise ValueError(f"Network error occurred: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")


async def main():
    api_url = "http://localhost:8000/yolo-detect"
    payload = {
        "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTjB8OZF0Jv2LmfP2GgWTUdZGmcIzXcV3kyUg&s"
    }

    try:
        for i in range(100):
            a = await send_request_to_endpoint(api_url, payload)
            print(i)
            print(f"Response {i}: {a}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
