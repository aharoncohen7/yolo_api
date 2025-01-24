import json
import pytz
import logging
import asyncio
import aioboto3
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta


from modules import AlertsRequest, AlertsResponse, DetectionEncoder, YoloData
from services import YoloService, MaskService, APIService


class SQSService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance


class SQSService:
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

            # Enhanced metrics tracking
            self._metrics = {
                'total_run_time': datetime.now(),
                'receives': 0,
                'sends': 0,
                'no_motion': 0,
                'no_detection': 0,
                'no_detection_on_mask': 0,
                'expires': 0,
                'get_errors': 0,
                'send_errors': 0,
                'delete_errors': 0,
                'general_errors': 0,
            }

            # New metrics tracking
            self._processing_times = {
                'with_detection': [],
                'without_detection': [],
                'total': []
            }

            # Time between camera event and detection
            self._camera_to_detection_times = []

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

    async def send_message(self, detection_data: AlertsResponse) -> bool:
        try:
            sqs = await self.get_sqs_client()
            await sqs.send_message(
                QueueUrl=self.backend_queue_url,
                MessageBody=json.dumps(detection_data, cls=DetectionEncoder)
            )
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

    async def process_message(self, message: Dict):
        start_time = datetime.now()
        try:
            message_body: AlertsRequest = AlertsRequest(
                **json.loads(message['Body']))

            S3urls = message_body.snapshots
            camera_data = message_body.camera_data
            frames = [await APIService.fetch_image(url) for url in S3urls]

            if not np.any(frames):
                await self.delete_message(message['ReceiptHandle'])
                async with self._metrics_lock:
                    self._metrics['expires'] += 1
                    self._processing_times['without_detection'].append(
                        (datetime.now() - start_time).total_seconds())
                return

            mask = MaskService.create_combined_mask(
                frames[0].shape, camera_data.masks, camera_data.is_focus)

            if len(frames) > 1:
                if not MaskService.detect_significant_movement(frames, mask):
                    await self.delete_message(message['ReceiptHandle'])
                    async with self._metrics_lock:
                        self._metrics['no_motion'] += 1
                        self._processing_times['without_detection'].append(
                            (datetime.now() - start_time).total_seconds())
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
                detection = AlertsResponse(
                    camera_data=message_body.without_camera_data(),
                    detections=detections
                )
                async with self._metrics_lock:
                    camera_event_time = datetime.fromisoformat(
                        str(detection.camera_data.event_time))
                    detection_time = datetime.now(camera_event_time.tzinfo)
                    camera_to_detection_time = (
                        detection_time - camera_event_time).total_seconds()
                    self._metrics['sends'] += 1
                    self._processing_times['with_detection'].append(
                        (detection_time - start_time.astimezone(camera_event_time.tzinfo)).total_seconds())
                    self._camera_to_detection_times.append(
                        camera_to_detection_time)

                MaskService.print_results(detections)
                await self.send_message(detection)
            else:
                async with self._metrics_lock:
                    if (detection_result and any(detection_result)):
                        self._metrics['no_detection_on_mask'] += 1
                    else:
                        self._metrics['no_detection'] += 1
                    self._processing_times['without_detection'].append(
                        (datetime.now() - start_time).total_seconds())

            # Delete processed message
            await self.delete_message(message['ReceiptHandle'])

        except Exception as e:
            async with self._metrics_lock:
                self._metrics['general_errors'] += 1
            self.logger.error(f"Message processing error: {e}")

    async def get_metrics(self) -> Dict:
        async with self._metrics_lock:
            # Calculate base metrics
            total_run_time = datetime.now() - self._metrics['total_run_time']
            total_send_attempts = self._metrics['receives'] + \
                self._metrics['sends']
            total_errors = sum([
                self._metrics['send_errors'],
                self._metrics['delete_errors'],
                self._metrics['get_errors'],
                self._metrics['general_errors']
            ])

            # Prepare metrics dictionary
            metrics = {
                'total_run_time': self._format_time(total_run_time),
                'receives': self._metrics['receives'],
                'sends': self._metrics['sends'],
                'no_motion': self._metrics['no_motion'],
                'no_detection': self._metrics['no_detection'],
                'no_detection_on_mask': self._metrics['no_detection_on_mask'],
                'expires': self._metrics['expires'],
                'processing_times': {
                    'with_detection': {
                        'avg': self._format_time(timedelta(seconds=np.mean(self._processing_times['with_detection']) if self._processing_times['with_detection'] else 0)),
                        'median': self._format_time(timedelta(seconds=np.median(self._processing_times['with_detection']) if self._processing_times['with_detection'] else 0)),
                        'std': self._format_time(timedelta(seconds=np.std(self._processing_times['with_detection']) if self._processing_times['with_detection'] else 0))
                    },
                    'without_detection': {
                        'avg': self._format_time(timedelta(seconds=np.mean(self._processing_times['without_detection']) if self._processing_times['without_detection'] else 0)),
                        'median': self._format_time(timedelta(seconds=np.median(self._processing_times['without_detection']) if self._processing_times['without_detection'] else 0)),
                        'std': self._format_time(timedelta(seconds=np.std(self._processing_times['without_detection']) if self._processing_times['without_detection'] else 0))
                    }
                },
                'camera_to_detection_times': {
                    'avg': self._format_time(timedelta(seconds=np.mean(self._camera_to_detection_times) if self._camera_to_detection_times else 0)),
                    'median': self._format_time(timedelta(seconds=np.median(self._camera_to_detection_times) if self._camera_to_detection_times else 0)),
                    'std': self._format_time(timedelta(seconds=np.std(self._camera_to_detection_times) if self._camera_to_detection_times else 0))
                },
                'detection_rate': (
                    self._metrics['sends'] / total_send_attempts * 100
                    if total_send_attempts else 0
                ),
                'error_rate': (
                    total_errors / total_send_attempts * 100
                    if total_send_attempts else 0
                )
            }

        return metrics

    def _format_time(self, time_delta: timedelta) -> str:
        units = [('Y', 365*24*60*60), ('M', 30*24*60*60), ('D', 24*60*60),
                 ('h', 3600), ('m', 60), ('s', 1)]

        total_seconds = int(time_delta.total_seconds())

        result = []
        for unit in units:
            unit_qty = total_seconds // unit[1]
            if unit_qty > 0:
                result.append(f"{unit_qty}{unit[0]}")
                total_seconds -= unit_qty * unit[1]

        return ' '.join(result) if result else '0s'
