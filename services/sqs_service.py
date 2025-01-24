import json
import logging
import asyncio
import aioboto3
import numpy as np
from datetime import datetime, timedelta
from services import YoloService
from typing import List, Dict, Optional
from services.api_service import APIService
from services.mask_service import MaskService
from modules import AlertsRequest, AlertsResponse
from modules.models import DetectionEncoder, YoloData


class SQSService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        region: str,
        data_for_queue_url: str,
        backend_queue_url: str,
        yolo_service: YoloService,
        batch_size: int = 20,
    ):
        if not hasattr(self, 'initialized'):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.session = aioboto3.Session(region_name=region)
            self._sqs_client = None
            self.data_for_queue_url = data_for_queue_url
            self.backend_queue_url = backend_queue_url
            self.yolo = yolo_service
            self.batch_size = batch_size
            self._metrics_lock = asyncio.Lock()
            self._metrics = {
                'total_run_time': datetime.now(),
                'work_run_time': timedelta(),
                'no_motion': 0,
                'no_detection': 0,
                'no_detection_on_mask': 0,
                'expires': 0,
                'send_errors': 0,
                'delete_errors': 0,
                'get_errors': 0,
                'general_errors': 0,
                'receives': 0,
                'sends': 0,
            }
            self.time_process: Optional[datetime] = None
            self.initialized = True

    async def get_sqs_client(self):
        """
        Returns an active SQS client. Creates a new one if it does not exist or is closed.
        """
        try:
            if not hasattr(self, '_sqs_client') or self._sqs_client is None:
                async with self.session.client('sqs') as sqs:
                    self._sqs_client = sqs
            return self._sqs_client
        except Exception as e:
            self.logger.error(f"Error creating or retrieving SQS client: {e}")
            self._sqs_client = None
            raise

    async def get_messages(self) -> List[Dict]:
        try:
            sqs = await self.get_sqs_client()
            response = await sqs.receive_message(
                QueueUrl=self.data_for_queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5,
                VisibilityTimeout=10
            )
            messages = response.get('Messages', [])
            self._metrics['receives'] += len(messages)
            return messages
        except Exception as e:
            self.logger.error(f"SQS receive error: {e}")
            self._metrics['get_errors'] += 1
            return []

    async def send_message(self, detection_data: Dict) -> bool:
        try:
            sqs = await self.get_sqs_client()
            await sqs.send_message(
                QueueUrl=self.backend_queue_url,
                MessageBody=json.dumps(detection_data, cls=DetectionEncoder)
            )
            self._metrics['sends'] += 1
            return True
        except Exception as e:
            self.logger.error(f"SQS send error: {e}")
            self._metrics['send_errors'] += 1
            return False

    async def delete_message(self, receipt_handle: str) -> bool:
        try:
            sqs = await self.get_sqs_client()
            await sqs.delete_message(
                QueueUrl=self.data_for_queue_url,
                ReceiptHandle=receipt_handle
            )
            return True
        except Exception as e:
            self.logger.error(f"SQS delete error: {e}")
            self._metrics['delete_errors'] += 1
            return False

    async def process_message(self, message: Dict):
        try:
            start = datetime.now()
            message_body: AlertsRequest = AlertsRequest(
                **json.loads(message['Body']))
            S3urls = message_body.snapshots
            camera_data = message_body.camera_data
            frames = [await APIService.fetch_image(url) for url in S3urls]

            # Add authentication or pre-signed URL logic if needed
            if not np.any(frames):
                print("un valid frames retrieved -> type:", 'xml')
                await self.delete_message(message['ReceiptHandle'])
                async with self._metrics_lock:
                    self._metrics['expires'] += 1
                    self._metrics['work_run_time'] += datetime.now() - start
                return

            mask = MaskService.create_combined_mask(
                frames[0].shape, camera_data.masks, camera_data.is_focus)

            # Handle motion detection if needed
            if len(frames) > 1:
                if not MaskService.detect_significant_movement(frames, mask):
                    print('No motion')
                    await self.delete_message(message['ReceiptHandle'])
                    async with self._metrics_lock:
                        self._metrics['no_motion'] += 1
                        self._metrics['work_run_time'] += datetime.now() - \
                            start
                    return

            yolo_data = YoloData(
                image=frames,
                confidence=camera_data.confidence,
                classes=camera_data.classes
            )
            detection_result = await self.yolo.add_data_to_queue(yolo_data=yolo_data)
            detections = [MaskService.get_detections_on_mask(
                det, mask, frames[0].shape) for det in detection_result]
            if detections and any(detections):
                print('Object detect!!!')
                # Create detection response
                detection = AlertsResponse(
                    camera_data=message_body.without_camera_data(),
                    detections=detections
                )
                MaskService.print_results(detections)

                # # Send detection result
                await self.send_message(detection)
            else:
                print('No Object Detected')
                async with self._metrics_lock:
                    if (detection_result and any(detection_result)):
                        self._metrics['no_detection_on_mask'] += 1
                    else:
                        self._metrics['no_detection'] += 1

            # Delete processed message
            await self.delete_message(message['ReceiptHandle'])
            async with self._metrics_lock:
                self._metrics['work_run_time'] += datetime.now() - start

            # Add any processing logic here
        except Exception as e:
            async with self._metrics_lock:
                self._metrics['general_errors'] += 1
                self._metrics['work_run_time'] += datetime.now() - start
            self.logger.error(f"Message processing error: {e}")

    async def continuous_transfer(self, poll_interval: int = 0):
        while True:
            try:
                messages = await self.get_messages()
                print(f"Receive: {len(messages)} motions")
                if messages:
                    tasks = [self.process_message(msg) for msg in messages]
                    await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(poll_interval)

            except Exception as e:
                self.logger.error(f"Transfer loop error: {e}")
                await asyncio.sleep(poll_interval)


    def _format_time(self, time_delta: timedelta) -> str:
        units = [('y', 365*24*60*60), ('m', 30*24*60*60), ('d', 24*60*60),
                ('h', 3600), ('m', 60), ('s', 1)]

        total_seconds = int(time_delta.total_seconds())
        return ' '.join(f"{total_seconds // unit[1]}{unit[0]}"
                        for unit in units
                        if (qty := total_seconds // unit[1]) > 0)


    def get_metrics(self) -> Dict:
        metric = self._metrics.copy()
        now = datetime.now()

        metric['total_run_time'] = self._format_time(
            now - metric['total_run_time'])
        metric['work_run_time'] = self._format_time(metric['work_run_time'])
        metric['avg_detection_time'] = self._format_time(
            metric['work_run_time'] / metric['no_detection']
            if metric['no_detection'] else timedelta()
        )

        total_send_attempts = metric['receives'] + metric['sends']
        metric['success_rate'] = (
            metric['sends'] / total_send_attempts * 100) if total_send_attempts else 0

        total_errors = sum([
            metric['send_errors'],
            metric['delete_errors'],
            metric['get_errors'],
            metric['general_errors']
        ])
        metric['error_rate'] = (
            total_errors / total_send_attempts * 100) if total_send_attempts else 0

        metric['not_sends'] = metric['receives'] - metric['sends']
        return metric
