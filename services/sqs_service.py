import json
import logging
import asyncio
import aioboto3
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict
from pydantic import ValidationError

from services import YoloService, MaskService, S3Service
from modules import AlertsRequest, AlertsResponse, DetectionEncoder, YoloData, metrics_tracker


class SQSService:
    def __init__(
        self,
        region: str,
        data_for_queue_url: str,
        backend_queue_url: str,
        yolo_service: YoloService,
        S3Service: S3Service,
        batch_size: int = 20,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

        self.session = aioboto3.Session(region_name=region)
        self._sqs_client = None
        self.data_for_queue_url = data_for_queue_url
        self.backend_queue_url = backend_queue_url
        self.yolo = yolo_service
        self.S3Service = S3Service
        self.batch_size = batch_size

    async def get_sqs_client(self):
        try:
            if self._sqs_client is None:
                self._sqs_client = await self.session.client('sqs').__aenter__()
            return self._sqs_client
        except Exception as e:
            self.logger.error(
                "Error creating or retrieving SQS client", exc_info=True)
            self._sqs_client = None
            raise

    async def get_Alerts(self) -> List[Dict]:
        try:
            sqs = await self.get_sqs_client()
            response = await sqs.receive_message(
                QueueUrl=self.data_for_queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5,
                VisibilityTimeout=10
            )
            Alerts = response.get('Messages', [])
            await metrics_tracker.update('receives', len(Alerts))
            await metrics_tracker.update('Alert_in_action', len(Alerts))
            return Alerts
        except Exception as e:
            self.logger.error(
                "Failed to receive Alerts from SQS", exc_info=True)
            await metrics_tracker.update('errors', {'get': 1})
            return []

    async def send_Alert(self, detection_data: AlertsResponse) -> bool:
        try:
            sqs = await self.get_sqs_client()
            await sqs.send_message(
                QueueUrl=self.backend_queue_url,
                MessageBody=json.dumps(detection_data, cls=DetectionEncoder)
            )
            self.logger.info(f"✅ Successfully sent detection data")
            await metrics_tracker.update('sends')
            return True
        except Exception as e:
            self.logger.error("❌ Failed to send Alert to SQS", exc_info=True)
            await metrics_tracker.update('errors', {'send': 1})
            return False

    async def delete_Alert(self, receipt_handle: str) -> bool:
        try:
            sqs = await self.get_sqs_client()
            await sqs.delete_message(
                QueueUrl=self.data_for_queue_url,
                ReceiptHandle=receipt_handle
            )
            await metrics_tracker.update('Alert_in_action', -1)
            return True
        except Exception as e:
            self.logger.error(
                "❌ Failed to delete Alert from SQS", exc_info=True)
            await metrics_tracker.update('errors', {'delete': 1})
            return False

    async def process_Alert(self, Alert: Dict):
        start_time = datetime.now()
        try:
            try:
                Alert_body = AlertsRequest(**json.loads(Alert['Body']))
            except ValidationError as ve:
                self.logger.warning(
                    f"⚠️ Validation error for Alert", exc_info=True)
                await self.delete_Alert(Alert['ReceiptHandle'])
                return

            camera_data = Alert_body.camera_data
            frames = await asyncio.gather(*[self.S3Service.fetch_image(key=url) for url in Alert_body.snapshots], return_exceptions=True)

            if not all(isinstance(frame, np.ndarray) for frame in frames):
                self.logger.warning(f"❌ Expired or invalid image URLs")
                await self.delete_Alert(Alert['ReceiptHandle'])
                await metrics_tracker.update('expires')
                detection_happened = False
                await metrics_tracker.add_processing_time((datetime.now() - start_time).total_seconds(), detection_happened)
                return

            mask = MaskService.create_combined_mask(
                frames[0].shape, camera_data.masks, camera_data.is_focus)

            # if len(frames) > 1 and isinstance(mask, np.ndarray):
            if len(frames) > 1:
                test_mask_time = datetime.now()
                is_def, mask, color_mask = MaskService.detect_significant_movement(
                    frames, mask, mask_with_movement=True)
                await metrics_tracker.add_detect_motion_time(
                    (datetime.now() - test_mask_time).total_seconds())
                if not is_def:
                    self.logger.info(f"🛑 No significant movement detected")
                    await self.delete_Alert(Alert['ReceiptHandle'])
                    await metrics_tracker.update('no_motion')
                    detection_happened = False
                    await metrics_tracker.add_processing_time((datetime.now() - start_time).total_seconds(), detection_happened)
                    return
            # for frame in frames:
            #     cv2.imshow("Slideshow", frame)
            #     cv2.waitKey(500)
            # cv2.destroyAllWindows()

            # cv2.imwrite("color_mask.jpg", color_mask)
            # input("stop to view the color_mask:")

            yolo_data = YoloData(
                image=frames, confidence=camera_data.confidence, classes=camera_data.classes)
            detection_result = await self.yolo.add_data_to_queue(yolo_data=yolo_data)
            # detections_mask = [MaskService.get_detections_on_mask(
            #     det, mask, frames[0].shape) for det in detection_result]
            detections = [MaskService.get_detections_on_mask(
                det, mask, frames[0].shape) for det in detection_result]

            detection_happened = False
            if detections and any(detections):
                mask_key = f'{Alert_body.snapshots[0][:-6]}_3.jpg'
                await self.S3Service.upload_image(key=mask_key, image=color_mask)
                Alert_body.snapshots.append(mask_key)
                detection = AlertsResponse(
                    camera_data=Alert_body.without_camera_data(), detections=detections)
                MaskService.print_results(detections)
                await self.send_Alert(detection)
                detection_happened = True
            else:
                if detection_result and any(detection_result):
                    self.logger.info(f"🕵️ Detection outside the mask")
                    await metrics_tracker.update('no_detection_on_mask')
                else:
                    self.logger.info(f"🚶 Movement but No detection")
                    await metrics_tracker.update('no_detection')

            await self.delete_Alert(Alert['ReceiptHandle'])
            await metrics_tracker.process_detection_time(Alert_body.event_time, start_time)

        except Exception as e:
            await metrics_tracker.update('errors', {'general': 1})
            await metrics_tracker.update('Alert_in_action', -1)
            self.logger.error(f"❌ Error processing Alert: {Alert.get('MessageId', 'Unknown ID')}", exc_info=True)
            await self.delete_Alert(Alert['ReceiptHandle'])

    async def continuous_transfer(self, poll_interval: int = 0):
        while True:
            try:
                Alerts = await self.get_Alerts()
                if len(Alerts) > 0:
                    self.logger.info(
                        f"📬 Processing {len(Alerts)} Alerts")
                if Alerts:
                    tasks = [self.process_Alert(msg) for msg in Alerts]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in results:
                        if isinstance(result, Exception):
                            self.logger.error(
                                "❌ Error while processing a Alerts", exc_info=result)

                await asyncio.sleep(poll_interval)

            except Exception as e:
                self.logger.error(
                    "❌ Error in continuous transfer loop", exc_info=True)
                await asyncio.sleep(poll_interval)

    async def get_metrics(self) -> Dict:
        return metrics_tracker.calculate_metrics()
