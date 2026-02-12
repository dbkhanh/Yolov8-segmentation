import json
from pathlib import Path

json_dir = Path("../photo/labels_json")
output_dir = Path("../photo/labels")
output_dir.mkdir(exist_ok=True)

class_map = {
    "bag": 0
}

for json_path in json_dir.glob("*.json"):
    with open(json_path) as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        class_id = class_map[label]

        points = shape["points"]
        normalized = []

        for x, y in points:
            normalized.append(x / img_w)
            normalized.append(y / img_h)

        line = str(class_id) + " " + " ".join(f"{v:.6f}" for v in normalized)
        lines.append(line)

    txt_path = output_dir / f"{json_path.stem}.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✔ Converted {json_path.name}")
