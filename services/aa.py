def convert_bbox(bbox: dict[str, float], mask: np.ndarry) -> bool:

    x1, x2, y1, y2 = bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]

    points = [
        (int(x), int(y))
        for i, x in enumerate([x1, (x1 + x2) // 2, x2])
        for j, y in enumerate([y1, (y1 + y2) // 2, y2])
        # שולל קצוות
        if not (i + j) % 2 == 0
    ]

    return any(mask[y, x] > 0 for x, y in points)
