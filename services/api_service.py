import cv2
import aiohttp
import numpy as np
from typing import Optional


class APIService:
    @staticmethod
    async def fetch_image(url: str) -> Optional[np.ndarray] | False:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        content_type = response.headers.get(
                            'Content-Type', '').lower()
                        if 'xml' in content_type:
                            return False
                        raise ValueError(f"Failed to fetch image: {response.status}, {await response.text()}")
                    image_data = await response.read()

                    nparr = np.frombuffer(image_data, np.uint8)

                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("Failed to decode image data.")
                    return img

        except aiohttp.ClientError as e:
            raise ValueError(f"Network error occurred: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {str(e)}")
