import csv
import os
from PIL import ImageDraw
from transformers import TableTransformerForObjectDetection
from transformers import AutoModelForObjectDetection
from torchvision import transforms
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch


from django.conf import settings


from table_extractor.utils import (
    MaxResize,
    objects_to_crops,
    outputs_to_objects,
    get_cell_coordinates_by_row,
    apply_ocr,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TABLE_DETECTION_MODEL = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection", revision="no_timm"
)
TABLE_STRUCTURE_RECOGNITION_MODEL = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-structure-recognition-v1.1-all"
)


TABLE_DETECTION_MODEL.to(DEVICE)
TABLE_STRUCTURE_RECOGNITION_MODEL.to(DEVICE)

DETECTION_TRANSFORM = transforms.Compose(
    [
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
STRUCTURE_TRANSFORM = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

ID2LABEL = TABLE_DETECTION_MODEL.config.id2label
ID2LABEL[len(TABLE_DETECTION_MODEL.config.id2label)] = "no object"

STRUCTURE_ID2LABEL = TABLE_STRUCTURE_RECOGNITION_MODEL.config.id2label
STRUCTURE_ID2LABEL[len(STRUCTURE_ID2LABEL)] = "no object"


def get_table_from_image(image):
    """
    Extracts the first detected table from an image using a pre-trained deep learning model.

    Args:
        image (PIL.Image): The input image containing a table. It's assumed to be
                           in RGB format.

    Returns:
        PIL.Image: The extracted table image, cropped from the original image,
                   or None if no table is detected.

    """
    pixel_values = DETECTION_TRANSFORM(image).unsqueeze(0)
    pixel_values = pixel_values.to(DEVICE)

    with torch.no_grad():
        outputs = TABLE_DETECTION_MODEL(pixel_values)

    detection_class_thresholds = {"table": 0.5, "table rotated": 0.5, "no object": 10}
    objects = outputs_to_objects(outputs, image.size, ID2LABEL)

    tables_crops = objects_to_crops(
        image, objects, detection_class_thresholds, padding=0
    )
    return tables_crops[0]["image"].convert("RGB")


def get_table_cells_coordinates(cropped_table):
    """
    Extracts the coordinates of cells within a cropped table image.

    Args:
        cropped_table (PIL.Image): The cropped table image.

    Returns:
        list: A list of dictionaries, where each dictionary represents a cell
        and contains its coordinates.
            Each dictionary has the following keys:
                'row': The row index of the cell.
                'column': The column index of the cell.
                'x_min': The minimum x-coordinate of the cell bounding box.
                'y_min': The minimum y-coordinate of the cell bounding box.
                'x_max': The maximum x-coordinate of the cell bounding box.
                'y_max': The maximum y-coordinate of the cell bounding box.
    """

    pixel_values = STRUCTURE_TRANSFORM(cropped_table).unsqueeze(0)
    pixel_values = pixel_values.to(DEVICE)

    with torch.no_grad():
        outputs = TABLE_STRUCTURE_RECOGNITION_MODEL(pixel_values)

    cells = outputs_to_objects(outputs, cropped_table.size, STRUCTURE_ID2LABEL)
    return get_cell_coordinates_by_row(cells)


def convert_detected_table_to_csv(
    cropped_table, cell_coordinates, output_dir, min_accuracy=0.4
):
    """
    Converts a detected table to a CSV file.

    Args:
        cropped_table (PIL.Image): The cropped table image.
        cell_coordinates (list): A list of dictionaries representing the coordinates of cells within the table.
            Each dictionary contains keys: 'row', 'column', 'x_min', 'y_min', 'x_max', 'y_max'.
        output_dir (str): The directory where the CSV file will be saved.
        min_accuracy (float, optional): The minimum accuracy threshold for OCR. Defaults to 0.4.

    Returns:
        str: The path to the generated CSV file.
    """

    data = apply_ocr(cropped_table, cell_coordinates, min_accuracy)
    csv_path = os.path.join(output_dir, "output.csv")
    with open(csv_path, "w") as result_file:
        wr = csv.writer(result_file, dialect="excel")

        for row, row_text in data.items():
            wr.writerow(row_text)

    return csv_path
