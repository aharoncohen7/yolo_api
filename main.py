import cv2
import uvicorn
import numpy as np
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Query

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


@app.post("/yolo-detect/single", response_model=AlertResponse)
async def check_single_picture(request: AlertRequest):
    """
    Check single picture
    --------------------
    Endpoint to process a single image and detect objects using YOLO.

    Args:
    -----
    - request `AlertRequest`: Request data containing the image URL and camera data.
        - url `str`: URL of the image to be processed.
        - camera_data `CameraData`: Contains the configuration for detection, including:
            - confidence threshold `float` - goes from 0 to 1 when 1 is the highest,
            - class `list` [0,1,2,...80],

    Returns:
    --------
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
async def check_single_picture_with_mask(request: AlertRequest):
    """
    Check single picture with mask
    ------------------------------
    Endpoint to process a single image with a mask and detect objects using YOLO.

    Args:
    -----
    - request `AlertRequest`: Request data containing the image URL and camera data.
        - url `str`: URL of the image to be processed.
        - camera_data `CameraData`: Contains the configuration for detection, including:
            - confidence threshold `float` - goes from 0 to 1 when 1 is the highest,
            - class `list` [0,1,2,...80],
            - masks `list[Shape]`: List of shapes the contain list of x,y `Coordinates`
            - is_focus status `bool` if the coordinates in masks are the main area the focus will be `True`.

    Returns:
    --------
    - `AlertResponse`: The detection results including URL, camera data, and filtered detections on the mask.
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
async def check_many_pictures(request: AlertsRequest):
    """
    Check many pictures
    -------------------
    Endpoint to process multiple images and detect objects using YOLO.

    Args:
    -----
    - request `AlertsRequest`: Request data containing the image URL and camera data.
        - urls `str`: URL of the image to be processed.
        - camera_data `CameraData`: Contains the configuration for detection, including:
            - confidence threshold `float` - goes from 0 to 1 when 1 is the highest,
            - class `list` [0,1,2,...80],

    Returns:
    --------
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
async def check_many_pictures_with_mask(request: AlertsRequest):
    """
    Check many pictures with mask
    -----------------------------

    Endpoint to process multiple images with a mask and detect objects using YOLO.

    Args:
    -----
    - request `AlertsRequest`: Request data containing the image URL and camera data.
        - urls `str`: URL of the image to be processed.
        - camera_data `CameraData`: Contains the configuration for detection, including:
            - confidence threshold `float` - goes from 0 to 1 when 1 is the highest,
            - class `list` [0,1,2,...80],
            - masks `list[Shape]`: List of shapes the contain list of x,y `Coordinates`
            - is_focus status `bool` if the coordinates in masks are the main area the focus will be `True`.

    Returns:
    --------
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
async def check_many_pictures_with_mask_and_motion(request: AlertsRequest):
    """
    check many pictures with mask and motion
    ----------------------------------------
    Endpoint to process multiple images with a mask and detect objects using YOLO, including motion detection.

    Args:
    -----
    - request `AlertsRequest`: Request data containing the image URL and camera data.
        - urls `str`: URL of the image to be processed.
        - camera_data `CameraData`: Contains the configuration for detection, including:
            - confidence threshold `float` - goes from 0 to 1 when 1 is the highest,
            - class `list` [0,1,2,...80],
            - masks `list[Shape]`: List of shapes the contain list of x,y `Coordinates`
            - is_focus status `bool` if the coordinates in masks are the main area the focus will be `True`.

    Returns:
    --------
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
async def generic_detection(request: Optional[AlertsRequest | AlertRequest], motion: Optional[bool] = Query("true")):
    """
    generic detection
    -----------------
    Endpoint to process one or more images and detect objects using YOLO, with optional motion detection.

    Args:
    -----
    - request `Optional[AlertsRequest | AlertRequest]`: 
        Request data containing one or more image URLs, camera configuration, and optional mask data.

        - If `AlertRequest`:
            - url: `str` URL of a single image to be processed.
            - camera_data: `CameraData` Contains the configuration for detection, including:
                - confidence: `float` Confidence threshold (0 to 1, where 1 is the highest).
                - classes: `list[int]` List of class IDs to detect (e.g., `[0, 1, 2, ..., 80]`).
                - masks: `list[Shape]` List of shapes the contain list of x,y `Coordinates`
                - is_focus: `bool` Indicates whether the area defined by the masks is the main focus (`True`).

        - If `AlertsRequest`:
            - urls: `list[str]` List of URLs for multiple images to be processed.
            - camera_data: `CameraData`: Same structure as above.

    - motion `Optional[bool]`: Query parameter to enable or disable motion detection. Defaults to `True`.


    Returns:
    --------
    - `AlertsResponse`: 
        Detection results containing:
        - `urls` (`list[str]`): Processed image URLs.
        - `camera_data` (`CameraData`): Configuration data used for detection.
        - `detections` (`list[list[Detection]]`): for each frame could be many Detection's objects with bounding boxes and some more details.

    - `Dict[str, str]`: 
        A message if no significant movement is detected when motion detection is enabled. 
        - For example: `{ "message": "No significant movement detected between frames..."}`

    Exceptions:
    -----------
    - Raises `HTTPException` with status 400 if image decoding fails.
    - Raises `HTTPException` with status 500 for any other internal error.
    """
    camera_data: CameraData = request.camera_data
    urls = [request.url] if hasattr(request, 'url') and isinstance(
        request.url, str) else request.urls

    try:
        # Fetch image from URL(s)
        frames = [await APIService.fetch_image(url) for url in urls]

        # For development testing
        # frames = [cv2.imread(url).astype(np.uint8()) for url in urls]

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
