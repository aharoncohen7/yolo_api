import cv2
import shutil
import numpy as np
from typing import List

from modules import Detection, Shape, Xyxy


class MaskService:

    @staticmethod
    def create_combined_mask(image_shape: tuple, masks: List[Shape], is_focus: bool = False) -> np.ndarray:
        """
        Creates a combined mask by merging multiple masks into one binary mask.

        :param image_shape: Shape of the image (height, width).
        :param masks: List of masks to combine.
        :param is_focus: If True, keeps focus areas; if False, keeps exclusion areas.
        :return: Combined binary mask (numpy array).
        """
        if masks:
            # Initialize an empty mask of the same shape as the image
            mask_regions = np.zeros(image_shape[:2], dtype=np.uint8)

            for mask_config in masks:
                # Create a temporary mask for each individual mask in the list
                temp_mask = np.zeros(image_shape[:2], dtype=np.uint8)
                points = [
                    [int(coord.x * image_shape[1]),
                     int(coord.y * image_shape[0])]
                    for coord in mask_config.shape
                ]
                points = np.array(points, dtype=np.int32)
                # Fill the polygon defined by the points into the temporary mask
                cv2.fillPoly(temp_mask, [points], 1)
                # Combine the mask with the existing one using OR operation
                mask_regions = cv2.bitwise_or(mask_regions, temp_mask)

            # Return the mask based on the 'is_focus' flag
            combined_mask = mask_regions if is_focus else 1 - mask_regions
            print(combined_mask)
        else:
            combined_mask = []

        return combined_mask

    @staticmethod
    def get_detections_on_mask(
        detections: List[Detection],
        mask: np.ndarray,
        shape: List[int]
    ) -> List[Detection]:
        """
        Filters detections based on whether their center is inside the provided mask.

        :param detections: List of detection objects (bounding boxes).
        :param mask: Binary mask to check against.
        :return: List of detections whose center is inside the mask.
        """
        if isinstance(mask, np.ndarray):
            ret_detections = []

            for detect in detections:
                # Calculate the center of each detection
                center_x = int((detect.bbox.x1 + detect.bbox.x2) / 2)
                center_y = int((detect.bbox.y1 + detect.bbox.y2) / 2)

                # If the center is inside the mask, keep the detection
                if mask[center_y, center_x] == 0:
                    continue

                # Convert bounding box to relative coordinates based on mask size
                detect.bbox = MaskService._convert_pixels_relative_to_image_shape(
                    detect.bbox,
                    shape[1],
                    shape[0]
                )
                ret_detections.append(detect)

            return ret_detections
        else:
            return MaskService.convert_bbox(detections, shape)

    @staticmethod
    def convert_bbox(detections: List[Detection], shape: List[int]) -> List[Detection]:
        """
        Converts the bounding box coordinates of detections to relative values based on image shape.

        :param detections: List of detection objects.
        :param shape: Image shape (height, width).
        :return: List of detections with converted bounding boxes.
        """
        ret_detections = []

        for detect in detections:
            # Convert the bounding box to relative coordinates
            detect.bbox = MaskService._convert_pixels_relative_to_image_shape(
                detect.bbox,
                shape[1],
                shape[0]
            )
            ret_detections.append(detect)

        return ret_detections

    @staticmethod
    def _convert_pixels_relative_to_image_shape(bbox: Xyxy, image_x: int, image_y: int):
        """
        Converts bounding box coordinates from pixel values to relative values.

        :param bbox: Bounding box in pixel coordinates.
        :param image_x: Width of the image.
        :param image_y: Height of the image.
        :return: Normalized bounding box in relative coordinates.
        """
        return Xyxy(
            x1=round(bbox.x1 / image_x, 3),
            x2=round(bbox.x2 / image_x, 3),
            y1=round(bbox.y1 / image_y, 3),
            y2=round(bbox.y2 / image_y, 3)
        )

    @staticmethod
    def accumulate_motion(prev_diff, current_diff, alpha=0.5):
        """
        Accumulates motion between two frames by blending their differences.

        :param prev_diff: The previous frame's difference.
        :param current_diff: The current frame's difference.
        :param alpha: Weight for the previous frame's difference in the blend.
        :return: Accumulated motion.
        """
        if prev_diff.shape != current_diff.shape:
            raise ValueError(
                "Shape mismatch between previous and current differences.")
        # Blend the two frame differences
        return cv2.addWeighted(prev_diff.astype(np.float32), alpha, current_diff.astype(np.float32), 1 - alpha, 0)

    @staticmethod
    def is_contour_in_mask(contour, mask):
        """
        Checks if a contour's center is inside the provided mask.

        :param contour: The contour to check.
        :param mask: The mask to check against.
        :return: True if the contour's center is inside the mask, False otherwise.
        """
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return False
        # Find the center of the contour
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        # Check if the center is within the mask bounds and inside the mask
        if mask and 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
            return mask[cy, cx] == 1
        return False or isinstance(mask, list)

    @staticmethod
    def detect_significant_movement(frames: List[np.ndarray], mask: np.ndarray, sensitivity: float = 1.0, min_area: int = 500) -> bool:
        """
        Detects significant motion in a series of frames using a mask.

        :param frames: List of frames to analyze (at least two).
        :param mask: Binary mask to apply.
        :param sensitivity: Motion sensitivity.
        :param min_area: Minimum area to consider significant motion.
        :return: True if significant motion is detected, False otherwise.
        """
        if len(frames) < 2:
            raise ValueError("At least two frames are required.")

        frame_shape = frames[0].shape
        # Ensure all frames have the same shape
        if not all(frame.shape == frame_shape for frame in frames):
            raise ValueError("All frames must have the same shape.")

        # Ensure mask shape matches frame shape
        if isinstance(mask, np.ndarray) and mask.shape[:2] != frame_shape[:2]:
            raise ValueError("Mask shape must match frame shape.")

        threshold = 25 + int(sensitivity * 50)
        blur_kernel_size = max(int(15 * sensitivity), 3)
        min_area = int(min_area * (1 - sensitivity))

        motion_accumulator = np.zeros(frame_shape[:2], dtype=np.float32)

        for i in range(len(frames) - 1):
            # Calculate the frame difference
            diff = cv2.absdiff(frames[i], frames[i + 1])
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to the difference
            blurred_diff = cv2.GaussianBlur(
                diff_gray, (blur_kernel_size, blur_kernel_size), 0)
            blurred_diff = cv2.medianBlur(blurred_diff, 5)
            _, binary_diff = cv2.threshold(
                blurred_diff, threshold, 255, cv2.THRESH_BINARY)

            # Accumulate motion over all frame differences
            motion_accumulator = MaskService.accumulate_motion(
                motion_accumulator, binary_diff)

        # Find contours in the accumulated motion mask
        contours, _ = cv2.findContours(motion_accumulator.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        # Filter contours based on whether they are inside the mask
        filtered_contours = [
            cnt for cnt in significant_contours if MaskService.is_contour_in_mask(cnt, mask)]

        # Return True if significant movement was detected
        return len(filtered_contours) > 0

    @staticmethod
    def print_results(detections: List[Detection] | List[List[Detection]], shape: tuple[int] = (10, 16, 16, 40)):
        """
        Prints the results of object detections in a formatted table.

        Args:
            detections (List[Detection] | List[List[Detection]]): A list of detections, either as a flat list or 
                a list of lists, where each element is an instance of the `Detection` class. Each detection should
                contain information such as class name, confidence, and bounding box.
            shape (tuple[int], optional): A tuple specifying the width of the columns in the printed table. 
                Defaults to (10, 20, 20, 40), representing the widths of the Before table columns for Class Name, Confidence, and 
                Bounding Box respectively.

        """

        # Check if detections are empty
        if not detections or (isinstance(detections[0], list) and not any(detections)):
            print("\n\n\033[91m   - No detections found -   \033[0m\n\n")
            return

        # If detections is not a list of lists, convert it to one
        if not isinstance(detections[0], list):
            detections = [detections]

        # Check terminal width
        width = shutil.get_terminal_size().columns - 4
        B, F, S, L = shape
        B = int((width//2)-((F+S+L)//2)-10) if width > F+S+L else 1

        # Define the table headers and divider
        headers = f"{
            " " * B}| {'Class Name':^{F}}|{'Confidence':^{S}}|{'bbox':^{L}}|"
        divider = f"{" " * B}|-{'-'*F}|{'-'*S}|{'-'*L}|"

        # Print the table
        print("\n\n" + divider)
        print(headers)
        print(divider)

        # Loop through each set of detections (each image)
        for i, image_detections in enumerate(detections, 1):
            # Print the image header (no detections shown yet)
            print(f"{" " * B}| {f'image {i}':<{F}}|{' ' * S}|{' ' * L}|")

            # Print each detection's details
            for j, det in enumerate(image_detections, 1):
                print(f"{" " * B}| {f'{j}. {det.class_name}':^{F}}|{
                      f'{det.confidence:.3f} %':^{S}}|{f'{det.bbox}':^{L}}|")

            print(divider)

        print()

    # @staticmethod
    # def print_results(detections: List[Detection] | List[List[Detection]], shape: tuple[int] = (10, 16, 16, 40)):
    #     """
    #     Prints the results of object detections in a formatted table.

    #     Args:
    #         detections (List[Detection] | List[List[Detection]]): A list of detections, either as a flat list or
    #             a list of lists, where each element is an instance of the `Detection` class.
    #         shape (tuple[int], optional): Width of the columns in the table. Defaults to (10, 16, 16, 40).
    #     """
    #     # Colors
    #     RESET = "\033[0m"   # Reset
    #     CYA = "\033[36m"    # Cyan (Key)
    #     WHT = "\033[37m"    # White (Value)

    #     # Bright colors
    #     B_RED = "\033[91m"  # Bright Red
    #     B_GRE = "\033[92m"  # Bright Green
    #     B_YEL = "\033[93m"  # Bright Yellow
    #     B_BLU = "\033[94m"  # Bright Blue
    #     B_CYA = "\033[96m"  # Bright Cyan

    #     # Check if detections are empty
    #     if not detections:
    #         print(f"\n\n   {B_RED}- No detections found -{RESET}   \n\n")
    #         return

    #     # If detections is not a list of lists, convert it to one
    #     if not isinstance(detections[0], list):
    #         detections = [detections]

    #     # Check if all detection lists are empty
    #     elif not any(detections):
    #         print(f"\n\n   {B_RED}- No detections found -{RESET}   \n\n")
    #         return

    #     # Adjust table dimensions
    #     width = shutil.get_terminal_size().columns - 4
    #     B, F, S, L = shape
    #     B = int((width // 2) - ((F + S + L) // 2) -
    #             10) if width > F + S + L else 1

    #     # Define the table headers and divider
    #     headers = f"{' ' * B}| {B_BLU}{'Class Name':^{F}}{RESET}|{B_BLU}{
    #         'Confidence':^{S}}{RESET}|{B_BLU}{'bbox':^{L}}{RESET}|"
    #     divider = f"{' ' * B}|-{'-' * F}|{'-' * S}|{'-' * L}|"

    #     # Print the table
    #     print("\n\n" + divider)
    #     print(headers)
    #     print(divider)

    #     # Loop through each set of detections (each image)
    #     for i, image_detections in enumerate(detections, 1):
    #         # Print the image header with custom colors
    #         print(
    #             f"{' ' * B}| {B_BLU}{f'Image {i}':<{F}}{RESET}|{' ' * S}|{' ' * L}|")

    #         # Print each detection's details
    #         for j, det in enumerate(image_detections, 1):
    #             # Choose color based on confidence
    #             conf_color = (
    #                 B_GRE if det.confidence > 0.8 else  # High confidence
    #                 B_YEL if det.confidence > 0.5 else  # Medium confidence
    #                 B_RED                              # Low confidence
    #             )

    #             # Extract bbox details for separate coloring
    #             bbox_keys = [f"x1", "y1", "x2", "y2"]
    #             bbox_values = det.bbox

    #             formatted_bbox = " ".join(
    #                 f"{CYA}{key}{RESET}={B_CYA}{value[1]:.3f}{RESET}" for key, value in zip(bbox_keys, bbox_values)
    #             )

    #             # Print detection details
    #             print(
    #                 f"{' ' * B}| {B_CYA}{f'{j}. {det.class_name}':^{F}}{RESET}|"
    #                 f"{conf_color}{f'{det.confidence:.3f}':^{S}}{RESET}|"
    #                 f"   {f'{formatted_bbox}':^{L}}  |"
    #             )

    #         # Print the divider after each image's detections
    #         print(divider)

    #     print()
