import uvicorn
import asyncio
from typing import Dict, Optional, Union
from fastapi import FastAPI, HTTPException, Query

from modules import AlertsResponse, AlertsRequest, YoloData
from services import APIService, YoloService, MaskService, SQSService

# Initialize FastAPI app
app = FastAPI(title="YOLO Detection Service")

# Initialize YoloService
yolo_service = YoloService()

# Initialize SqsService
SqsService = SQSService(
    region='il-central-1',
    data_for_queue_url='https://sqs.il-central-1.amazonaws.com/182399687196/ILG-motion-data-for-yolo',
    backend_queue_url='https://sqs.il-central-1.amazonaws.com/182399687196/ILG-object-detect-for-backend',
    yolo_service=yolo_service,
)


@app.on_event("startup")
async def startup_event():
    """
    Initialize YOLO and SQS services when the app starts.
    """
    await yolo_service.initialize()
    asyncio.create_task(SqsService.continuous_transfer())

# @app.post("/yolo-detect/single", response_model=AlertResponse)
# async def check_single_picture(request: AlertRequest):
#     """
#     Process a single image and detect objects using YOLO.

#     Args:
#     - request: Image URL and optional camera configuration (confidence, classes).

#     Returns:
#     - AlertResponse: Detection results with URL, camera data, and detected objects.
#     """
#     camera_data: CameraData = request.camera_data
#     url: str = request.url

#     try:
#         # Fetch image from URL
#         image: np.ndarray = await APIService.fetch_image(url)

#         # Prepare YOLO data for detection
#         yolo_data: YoloData = YoloData(
#             image=image, confidence=camera_data.confidence, classes=camera_data.classes)

#         # Process image with YOLO
#         detections = await yolo_service.add_image_to_queue(yolo_data)

#         # Print results
#         MaskService.print_results(detections)

#         return AlertResponse(url=url, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/yolo-detect/single-with-mask", response_model=AlertResponse)
# async def check_single_picture_with_mask(request: AlertRequest):
#     """
#     Process a single image with a mask and detect objects using YOLO.

#     Args:
#     - request: Image URL and camera configuration (confidence, classes, masks, focus).

#     Returns:
#     - AlertResponse: Detection results with mask filtering.
#     """
#     camera_data: CameraData = request.camera_data
#     url: str = request.url

#     try:
#         # Fetch image from URL
#         image: np.ndarray = await APIService.fetch_image(url)

#         # Prepare YOLO data for detection
#         yolo_data: YoloData = YoloData(
#             image=image, confidence=camera_data.confidence, classes=camera_data.classes)

#         # Process image with YOLO
#         detections = await yolo_service.add_image_to_queue(yolo_data)

#         # Generate mask and filter detections
#         mask = MaskService.create_combined_mask(
#             image.shape, camera_data.masks, camera_data.is_focus)
#         detections = MaskService.get_detections_on_mask(
#             detections, mask, image.shape)

#         # Print results
#         MaskService.print_results(detections)

#         return AlertResponse(url=url, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/yolo-detect/group", response_model=AlertsResponse)
# async def check_many_pictures(request: AlertsRequest):
#     """
#     Process multiple images and detect objects using YOLO.

#     Args:
#     - request: List of image URLs and optional camera configuration (confidence, classes).

#     Returns:
#     - AlertsResponse: Detection results for all images.
#     """
#     camera_data: CameraData = request.camera_data
#     urls: List[str] = request.urls

#     try:
#         # Fetch images from URLs
#         frames = [await APIService.fetch_image(url) for url in urls]

#         # Prepare YOLO data for detection
#         yolo_data = YoloData(
#             image=frames, classes=camera_data.classes, confidence=camera_data.confidence)

#         # Process with YOLO
#         detections = await yolo_service.add_image_to_queue(yolo_data)

#         # Print results
#         MaskService.print_results(detections)

#         return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/yolo-detect/group-with-mask", response_model=Optional[AlertsResponse | dict])
# async def check_many_pictures_with_mask(request: AlertsRequest):
#     """
#     Process multiple images with a mask and detect objects using YOLO.

#     Args:
#     - request: List of image URLs and camera configuration (confidence, classes, masks, focus).

#     Returns:
#     - AlertsResponse: Detection results filtered by mask.
#     - dict: Error message if no masks provided.
#     """
#     if not request.camera_data.masks:
#         return {"message": "Masks must be provided."}

#     camera_data: CameraData = request.camera_data
#     urls: List[str] = request.urls

#     try:
#         # Fetch images from URLs
#         frames = [await APIService.fetch_image(url) for url in urls]

#         # Prepare YOLO data for detection
#         yolo_data = YoloData(
#             image=frames, classes=camera_data.classes, confidence=camera_data.confidence)

#         # Process with YOLO
#         detections = await yolo_service.add_image_to_queue(yolo_data)

#         # Generate mask and filter detections
#         mask = MaskService.create_combined_mask(
#             frames[0].shape, camera_data.masks, camera_data.is_focus)
#         detections = [MaskService.get_detections_on_mask(
#             det, mask, frames[0].shape) for det in detections]

#         # Print results
#         MaskService.print_results(detections)

#         return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/yolo-detect/group-with-mask-and-motion", response_model=Optional[AlertsResponse | dict])
# async def check_many_pictures_with_mask_and_motion(request: AlertsRequest):
#     """
#     Process multiple images with a mask and motion detection using YOLO.

#     Args:
#     - request: List of image URLs and camera configuration (confidence, classes, masks, focus).

#     Returns:
#     - AlertsResponse: Detection results with motion filtering.
#     - dict: Message if no motion detected.
#     """
#     camera_data: CameraData = request.camera_data
#     urls: List[str] = request.urls

#     try:
#         # Fetch images from URLs
#         frames = [await APIService.fetch_image(url) for url in urls]

#         # Generate mask
#         mask = MaskService.create_combined_mask(
#             frames[0].shape, camera_data.masks, camera_data.is_focus)

#         # Check for significant motion
#         movement_detected = MaskService.detect_significant_movement(
#             frames, mask)
#         if not movement_detected:
#             return {"message": "No significant motion detected."}

#         # Prepare YOLO data for detection
#         yolo_data = YoloData(
#             image=frames, classes=camera_data.classes, confidence=camera_data.confidence)

#         # Process with YOLO
#         detections = await yolo_service.add_image_to_queue(yolo_data)

#         # Filter detections
#         detections = [MaskService.get_detections_on_mask(
#             det, mask, frames[0].shape) for det in detections]

#         # Print results
#         MaskService.print_results(detections)

#         return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/yolo-detect", response_model=Optional[Union[AlertsResponse, Dict[str, str]]])
async def generic_detection(request: Optional[AlertsRequest | AlertRequest], motion: Optional[bool] = Query("true")):
    """
    Detect objects in images using YOLO, with optional motion detection.

    Args:
    - request: Request containing image URL(s) and camera configuration, (confidence, classes, masks, focus).
    - motion: Whether to enable motion detection (default: True).
    - example: `http://localhost/yolo-detect/?motion=true`

    Returns:
    - AlertsResponse: Detection results with URLs, camera settings, and bounding box data.
    - Dict[str, str]: Message if no significant motion is detected.

    Raises:
    - HTTPException: For image decoding or internal errors.
    """

    # Get URLs and camera data
    print('data received')
    urls = [request.url] if hasattr(request, 'url') else request.urls
    camera_data = request.camera_data

    try:
        # Fetch images
        frames = [await APIService.fetch_image(url) for url in urls]
        if not frames:
            raise HTTPException(
                status_code=400, detail="Failed to decode image")

        # Create combined mask
        mask = MaskService.create_combined_mask(
            frames[0].shape, camera_data.masks, camera_data.is_focus)

        # Handle motion detection if needed
        if len(frames) > 1 and bool(motion):
            if not MaskService.detect_significant_movement(frames, mask):
                return {"message": "No significant movement detected. Use '/yolo-detect?motion=false' for batch detection."}

        # Prepare YOLO data
        yolo_data = YoloData(
            image=frames, classes=camera_data.classes, confidence=camera_data.confidence)
        detections = await yolo_service.add_data_to_queue(yolo_data=yolo_data)

        # Adjust detections for mask
        detections = [MaskService.get_detections_on_mask(
            det, mask, frames[0].shape) for det in detections]

        # Return the detection results
        return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

    except Exception as e:
        print(f"Error processing alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def get_metric():
    metric = SqsService.get_metrics()
    return {"data": metric, "status": 'healthy'}


if __name__ == "__main__":
    # Run FastAPI application with Uvicorn server for  => Development
    # Run with `Gunicorn` on Ubuntu system\server for  => Production
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        workers=4,
        reload=True
    )
