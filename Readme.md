# Yolo API Service

A simple service to process images using YOLO object detection.

---

## Getting Started

### 1. Clone the Project

```bash
git clone https://github.com/David-Abravanel/yolo_api.git
cd yolo_api
```

### 2. Set up a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Service

```bash
python main.py
```

The service will start locally at `http://localhost:8000`.

---

## Request Example

You can test the API using tools like [Postman](https://www.postman.com/) or `curl`.

### Endpoint

- **Method**: `POST`
- **URL**: `http://localhost:8000/yolo-detect`
- **Request Body (JSON)**:

```json
{
  "url": "https://example.com/your-image-url.jpg",
  "camera_data": {
    "confidence": 0.5,
    "classes": [0, 2],
    "is_focus": false,
    "masks": [
      {
        "shape": [
          { "x": 0.1234, "y": 0.2134 },
          { "x": 0.9876, "y": 0.5543 },
          { "x": 0.1212, "y": 0.3333 }
        ]
      },
      {
        "shape": [
          { "x": 0.5432, "y": 0.6543 },
          { "x": 0.8765, "y": 0.1234 },
          { "x": 0.4321, "y": 0.7654 }
        ]
      }
    ]
  }
}
```

### Explanation of Parameters

#### `camera_data`

- **`confidence`** (`float`): Minimum confidence threshold for detection. Default: `0.2`.
- **`classes`** (`list[int]`): Object classes to detect. Default: `[0, 1, 2]` (e.g., person, bicycle, car).
- **`is_focus`** (`bool`): If `true`, the detection focuses on the areas defined by the masks. Default: `true`.
- **`masks`** (`list`): List of polygon shapes to define the focus areas. Default: an empty list.

#### Example Mask

Each mask is a list of `x, y` coordinates representing the vertices of a polygon.

---

## Example Response

```json
{
  "url": "https://example.com/your-image-url.jpg",
  "camera_data": {
    "confidence": 0.5,
    "classes": [0, 2],
    "is_focus": false,
    "masks": [
      {
        "shape": [
          { "x": 0.1234, "y": 0.2134 },
          { "x": 0.9876, "y": 0.5543 },
          { "x": 0.1212, "y": 0.3333 }
        ]
      },
      {
        "shape": [
          { "x": 0.5432, "y": 0.6543 },
          { "x": 0.8765, "y": 0.1234 },
          { "x": 0.4321, "y": 0.7654 }
        ]
      }
    ]
  },
  "detections": [
    {
      "bbox": { "x1": 0.156, "y1": 0.224, "x2": 0.432, "y2": 0.666 },
      "class_name": "person",
      "class_id": 0,
      "confidence": 0.85
    },
    {
      "bbox": { "x1": 0.554, "y1": 0.372, "x2": 0.812, "y2": 0.734 },
      "class_name": "car",
      "class_id": 2,
      "confidence": 0.78
    }
  ]
}
```

### Explanation of Response

- **`input`**: Contains the original input data that was received in the request.
  - **`url`**: The image URL provided in the request.
  - **`camera_data`**: The camera data from the request, including the detection parameters.
- **`detections`**: A list of detected objects, each containing:
  - **`class`**: The class of the detected object (e.g., "person", "car").
  - **`confidence`**: The confidence level of the detection.
  - **`bounding_box`**: The bounding box for the detected object in normalized coordinates (`x_min`, `y_min`, `x_max`, `y_max`).

---

## Troubleshooting

If the service doesn't start or behaves unexpectedly, ensure:

- You are using the correct Python version (e.g., Python 3.8+).
- All dependencies are installed correctly.
- The `url` in the request is valid and accessible.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
