import os

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def is_image(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in IMG_EXT


def classify_dataset(tmp_dir: str) -> str:
    entries = os.listdir(tmp_dir)
    entries_paths = [os.path.join(tmp_dir, e) for e in entries]
    print(f"[CLASSIFY] Entries found: {entries}")
    print(f"[CLASSIFY] Entry paths: {entries_paths}")

    # --- 1. CSV DATASET ---
    for e in entries:
        if e.lower().endswith(".csv"):
            return "csv"

    # --- 2. Check folder structures ---
    dirs = [p for p in entries_paths if os.path.isdir(p)]
    files = [p for p in entries_paths if os.path.isfile(p)]

    # Flat image dataset (images in root)
    if files and all(is_image(f) for f in files):
        return "images_flat"

    # Category-based dataset
    if len(dirs) >= 2:
        category_like = True
        for d in dirs:
            inner = os.listdir(d)
            if not inner:
                category_like = False
                break
            inner_paths = [os.path.join(d, x) for x in inner]
            if not all(os.path.isfile(x) and is_image(x) for x in inner_paths):
                category_like = False
                break

        if category_like:
            return "images_by_category"

    return "unknown"

def dataset_type_description(dataset_type: str) -> str:
    descriptions = {
        "csv": "Dataset containing CSV files.",
        "images_flat": "Dataset with images stored in a flat structure.",
        "images_by_category": "Dataset with images organized into category folders.",
        "unknown": "Dataset type could not be determined."
    }
    return descriptions.get(dataset_type, "No description available.")