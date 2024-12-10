# import os
# import hmac
import uvicorn
# import hashlib
import numpy as np
# import subprocess
from typing import Dict, Optional, Union
from fastapi import FastAPI, HTTPException, Query, Request

from services import APIService, YoloService, MaskService

from modules import (
    AlertsResponse,
    AlertResponse,
    AlertsRequest,
    AlertRequest,
    CameraData,
    YoloData,
)

# Initialize FastAPI app
app = FastAPI(title="YOLO Detection Service")

# Initialize YoloService
yolo_service = YoloService()


@app.on_event("startup")
async def startup_event():
    """
    Event handler to initialize YOLO service when the app starts.

    Initializes the YoloService for object detection when the FastAPI application starts.
    """
    await yolo_service.initialize()


# @app.post("/webhook")
# async def github_webhook(request: Request):
#     # Extract GitHub signature
#     github_signature = request.headers.get("X-Hub-Signature-256")
#     if not github_signature:
#         raise HTTPException(status_code=400, detail="Missing signature header")

#     webhook_secret = os.getenv('GITHUB_WEBHOOK_SECRET')  # From GitHub settings

#     # Read payload
#     payload = await request.body()

#     # Calculate expected signature
#     secret = webhook_secret.encode()
#     expected_signature = "sha256=" + hmac.new(
#         key=secret, msg=payload, digestmod=hashlib.sha256
#     ).hexdigest()

#     # Validate signature
#     if not hmac.compare_digest(github_signature, expected_signature):
#         raise HTTPException(status_code=403, detail="Invalid signature")

#     # Execute deployment steps
#     try:
#         # Pull latest changes
#         subprocess.run(["git", "pull", "origin", "master"], check=True)

#         # Activate virtual environment
#         subprocess.run(["source", "venv/bin/activate"], shell=True, check=True)

#         # Install dependencies
#         subprocess.run(
#             ["pip", "install", "-r", "requirements.txt"], shell=True, check=True)

#         # # Restart application
#         subprocess.run(["sudo", "systemctl", "restart",
#                        "yolo_api"], check=True)

#     except subprocess.CalledProcessError as e:
#         raise HTTPException(
#             status_code=500, detail=f"Deployment failed: {str(e)}")

#     return {"status": "Deployment successful"}


@app.post("/yolo-detect/single", response_model=AlertResponse)
async def post_alert(request: AlertRequest):
    """
    Endpoint to process a single image and detect objects using YOLO.

    Args:
    - request (AlertRequest): Request data containing the image URL and camera data.
      - camera_data (CameraData): Contains the configuration for detection, including confidence threshold and class list.
      - url (str): URL of the image to be processed.

    Returns:
    - AlertResponse: The detection results including URL, camera data, and detected objects.
    """
    camera_data: CameraData = request.camera_data
    url: str = request.url

    try:
        # Fetch image from URL
        image: np.ndarray = await APIService.fetch_image(url)

        # Prepare YOLO data for detection
        yolo_data: YoloData = YoloData(
            image=image, confidence=camera_data.confidence, classes=camera_data.classes)

        # Process image with YOLO
        detections = await yolo_service.add_image_to_queue(yolo_data)

        # Custom print results
        MaskService.print_results(detections)

        return AlertResponse(url=url, camera_data=camera_data, detections=detections)

    except Exception as e:
        print(f"Error processing alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/yolo-detect/single-with-mask", response_model=AlertResponse)
async def post_alert(request: AlertRequest):
    """
    Endpoint to process a single image with a mask and detect objects using YOLO.

    Args:
    - request (AlertRequest): Request data containing the image URL and camera data.
      - camera_data (CameraData): Contains the configuration for detection, including confidence threshold, class list, masks, and focus status.
      - url (str): URL of the image to be processed.

    Returns:
    - AlertResponse: The detection results including URL, camera data, and filtered detections on the mask.
    """
    camera_data: CameraData = request.camera_data
    url: str = request.url

    try:
        # Fetch image from URL
        image: np.ndarray = await APIService.fetch_image(url)

        # Prepare YOLO data for detection
        yolo_data: YoloData = YoloData(
            image=image, confidence=camera_data.confidence, classes=camera_data.classes)

        # Process image with YOLO
        detections = await yolo_service.add_image_to_queue(yolo_data)

        # Generate binary mask
        mask = MaskService.create_combined_mask(
            image.shape, camera_data.masks, camera_data.is_focus)

        # Filter detection on mask and Normalize bbox coordinates to percentages relative to image dimensions
        detections = MaskService.get_detections_on_mask(
            detections, mask, image.shape)

        # Custom print results
        MaskService.print_results(detections)

        return AlertResponse(url=url, camera_data=camera_data, detections=detections)

    except Exception as e:
        print(f"Error processing alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/yolo-detect/group", response_model=AlertsResponse)
async def post_alert(request: AlertsRequest):
    """
    Endpoint to process multiple images and detect objects using YOLO.

    Args:
    - request (AlertsRequest): Request data containing URLs of images and camera data.
      - camera_data (CameraData): Contains the configuration for detection, including confidence threshold and class list.
      - urls (List[str]): List of URLs of the images to be processed.

    Returns:
    - AlertsResponse: The detection results including URLs, camera data, and detected objects.
    """
    camera_data: CameraData = request.camera_data
    urls: str = request.urls

    try:
        # Fetch image from URLs
        frames = [await APIService.fetch_image(url) for url in urls]

        # Prepare YOLO data for detection
        yolo_data = YoloData(
            image=frames, classes=camera_data.classes, confidence=camera_data.confidence)

        # Process with YOLO
        detections = await yolo_service.add_image_to_queue(yolo_data)

        # Custom print results
        MaskService.print_results(detections)

        return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

    except Exception as e:
        print(f"Error processing alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/yolo-detect/group-with-mask", response_model=Optional[AlertsResponse | dict])
async def post_alert(request: AlertsRequest):
    """
    Endpoint to process multiple images with a mask and detect objects using YOLO.

    Args:
    - request (AlertsRequest): Request data containing URLs of images, camera data, and masks.
      - camera_data (CameraData): Contains the configuration for detection, including confidence threshold, class list, masks, and focus status.
      - urls (List[str]): List of URLs of the images to be processed.

    Returns:
    - AlertsResponse: The detection results including URLs, camera data, and filtered detections on the mask.
    - dict: Error message if no masks are provided.
    """
    if not request.camera_data.masks:
        return {"message": "The masks must be send to this endpoint, you could send to '/yolo-detect/group' with no masks "}

    camera_data: CameraData = request.camera_data
    urls: str = request.urls

    try:
        # Fetch image from URLs
        frames = [await APIService.fetch_image(url) for url in urls]

        # Prepare YOLO data for detection
        yolo_data = YoloData(
            image=frames, classes=camera_data.classes, confidence=camera_data.confidence)

        # Process with YOLO
        detections = await yolo_service.add_image_to_queue(yolo_data=yolo_data)

        # Generate binary mask
        mask = MaskService.create_combined_mask(
            frames[0].shape, camera_data.masks, camera_data.is_focus)

        # Filter detection on mask and Normalize bbox coordinates to percentages relative to image dimensions
        detections = [MaskService.get_detections_on_mask(
            det, mask, frames[0].shape) for det in detections]

        # Custom print results
        MaskService.print_results(detections)

        return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

    except Exception as e:
        print(f"Error processing alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/yolo-detect/group-with-mask-and-motion", response_model=Optional[AlertsResponse | dict])
async def post_alert(request: AlertsRequest):
    """
    Endpoint to process multiple images with a mask and detect objects using YOLO, including motion detection.

    Args:
    - request (AlertsRequest): Request data containing URLs of images, camera data, and masks.
      - camera_data (CameraData): Contains the configuration for detection, including confidence threshold, class list, masks, and focus status.
      - urls (List[str]): List of URLs of the images to be processed.

    Returns:
    - AlertsResponse: The detection results including URLs, camera data, and filtered detections on the mask if significant motion is detected.
    - dict: Message if no significant motion is detected.
    """
    camera_data: CameraData = request.camera_data
    urls: str = request.urls

    try:
        # Fetch image from URLs
        frames = [await APIService.fetch_image(url) for url in urls]

        # Generate binary mask
        mask = MaskService.create_combined_mask(
            frames[0].shape, camera_data.masks, camera_data.is_focus)

        # Check movement on mask
        movement_detected = MaskService.detect_significant_movement(
            frames, mask)

        if movement_detected == False:
            return {"message": "No significant movement detected between frames."}

        # Prepare YOLO data for detection
        yolo_data = YoloData(
            image=frames, classes=camera_data.classes, confidence=camera_data.confidence)

        # Process with YOLO
        detections = await yolo_service.add_image_to_queue(yolo_data=yolo_data)

        # Filter detection on mask and Normalize bbox coordinates to percentages relative to image dimensions
        detections = [MaskService.get_detections_on_mask(
            det, mask, frames[0].shape) for det in detections]

        # Custom print results
        MaskService.print_results(detections)

        return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

    except Exception as e:
        print(f"Error processing alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/yolo-detect", response_model=Optional[Union[AlertsResponse, Dict[str, str]]])
async def post_alert(request: Optional[AlertsRequest | AlertRequest], motion: Optional[bool] = Query("true")):
    """
    Endpoint to process one or more images and detect objects using YOLO, with optional motion detection.

    Args:
    - request (Optional[AlertsRequest | AlertRequest]): Request data containing one or more image URLs, camera configuration, and optional mask data.
        - If `AlertRequest` is provided, the `url` of a single image is expected.
        - If `AlertsRequest` is provided, multiple `urls` of images are expected.
    - motion (Optional[bool]): Optional query parameter to enable or disable motion detection. Defaults to `True`.

    Returns:
    - AlertsResponse: The detection results including URLs, camera data, and detected objects.
    - Dict[str, str]: A message if no significant motion is detected (if motion is enabled).
    """
    camera_data: CameraData = request.camera_data
    urls = [request.url] if hasattr(request, 'url') and isinstance(
        request.url, str) else request.urls

    try:
        # Fetch image from URL(s)
        frames = [await APIService.fetch_image(url) for url in urls]
        if frames is None or []:
            raise HTTPException(
                status_code=400, detail="Failed to decode image")

        # Generate binary mask
        mask = MaskService.create_combined_mask(
            frames[0].shape, camera_data.masks, camera_data.is_focus)

        # If there's only one frame and motion detection is enabled, check for movement
        if len(frames) == 1:
            frames = frames[0]

        elif bool(motion):
            # Check movement on the mask
            movement_detected = MaskService.detect_significant_movement(
                frames, mask)

            if not movement_detected:
                return {"message": "No significant movement detected between frames. If you don't want to check motion and you have many pictures, send to this endpoint '/yolo-detect?motion=false' or '/yolo-detect/group' or '/yolo-detect/group-with-mask' if you have mask coordinates"}

        # Prepare YOLO data for detection
        yolo_data = YoloData(
            image=frames, classes=camera_data.classes, confidence=camera_data.confidence)

        # Process with YOLO
        detections = await yolo_service.add_image_to_queue(yolo_data=yolo_data)

        # Convert bounding box pixels to relative x, y percentages based on image size
        if detections and isinstance(detections[0], list):
            detections = [MaskService.get_detections_on_mask(
                det, mask, frames[0].shape) for det in detections]
        else:
            detections = [MaskService.get_detections_on_mask(
                detections, mask, frames.shape)]

        # Custom print results (optional)
        MaskService.print_results(detections)

        return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

    except Exception as e:
        print(f"Error processing alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    """
    Run the FastAPI application with Uvicorn server for development
    For production run with Gunicorn on ubuntu system or server
    """
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        workers=1,
        reload=True
    )
