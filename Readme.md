# Yolo Api Service

### start

- Clone the project
- open a virtual environment .venv
- Run pip install -r requirements.txt
- Run the main file

### Request Example

- go to postman and fill the fields

- Method: Post
- Endpoint: http://locahost/yolo-detect
- Request: ( body `json` )

```json
{
  "url": "https://example.com/...",
  "camera_data": {
    // default
    "confidence": 0.5, // = 0.2
    "classes": [0, 2], // = [0,1,2] -> person, bicycle, car,
    "is_focus": false, // = True -> main area
    "masks": [
      // = [] -> empty list
      {
        "shape": [
          {
            "x": 0.1234,
            "y": 0.2134
          },
          {
            "x": 0.9876,
            "y": 0.5543
          },
          {
            "x": 0.1212,
            "y": 0.3333
          }
        ]
      },
      {
        "shape": [
          {
            "x": 0.1234,
            "y": 0.2134
          },
          {
            "x": 0.9876,
            "y": 0.5543
          },
          {
            "x": 0.1212,
            "y": 0.3333
          }
        ]
      }
    ]
  }
}
```
