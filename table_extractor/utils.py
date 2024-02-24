import torch
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import os
import PyPDF2
from PIL import Image
import io
from transformers import TableTransformerForObjectDetection
import traceback
from pdf2image import convert_from_bytes

TABLE_STRUCTURE_RECOGNITION_MODEL = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-structure-recognition-v1.1-all"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TABLE_STRUCTURE_RECOGNITION_MODEL.to(DEVICE)

OCR_READER = easyocr.Reader(["en"])


class MaxResize(object):
    """
    A class that resizes an image to a maximum size.

    Attributes:
        max_size (int): The maximum size of the resized image.
    """

    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    """
    Converts a bounding box from center-size coordinates to corner coordinates.

    Args:
        x (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes
          in center-size coordinates (x_center, y_center, width, height).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) representing N bounding boxes in 
        corner coordinates (x_min, y_min, x_max, y_max).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    """
    Rescales a bounding box to the original image size.

    Args:
        out_bbox (torch.Tensor): A tensor of shape (N, 4) representing N
        bounding boxes in normalized coordinates (0-1).
        size (tuple): A tuple of two integers representing the original
        image size (width, height).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) representing N bounding boxes in pixel 
        coordinates.
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    """
    Converts the model outputs to a list of objects.

    Args:
        outputs (dict): A dictionary containing the model outputs, such as logits
        and pred_boxes.
        img_size (tuple): A tuple of two integers representing the original image 
        size (width, height).
        id2label (dict): A dictionary mapping the class ids to the class labels.

    Returns:
        list: A list of dictionaries, each containing the label, score, and bbox of an object.
    """
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects


def objects_to_crops(img, objects, class_thresholds, padding=10):
    """
    Extracts and processes bounding boxes from detected objects to generate
    cropped table images.

    Args:
        img (PIL.Image): The original image containing the tables.
        objects (list): A list of dictionaries representing detected objects,
                        each with keys like "label", "score", and "bbox".
        class_thresholds (dict): A dictionary mapping object labels to their
                                 minimum score thresholds for inclusion.
        padding (int, optional): The amount of padding to apply around
                                  bounding boxes (default: 10).

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              cropped table and contains the following key:
              - "image": The cropped PIL.Image of the table.

    Raises:
        ValueError: If an object label is not found in the class_thresholds dictionary.
    """

    table_crops = []
    for obj in objects:
        if obj["score"] < class_thresholds[obj["label"]]:
            continue

        cropped_table = {}

        bbox = obj["bbox"]
        bbox = [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        ]

        cropped_img = img.crop(bbox)

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj["label"] == "table rotated":
            cropped_img = cropped_img.rotate(270, expand=True)

        cropped_table["image"] = cropped_img
        table_crops.append(cropped_table)

    return table_crops


def plot_results(cells, class_to_visualize, cropped_table):
    """
    Plots the bounding boxes of cells of a specific class on a cropped table image.

    Args:
        cells (list): A list of dictionaries representing detected cells.
            Each dictionary contains information about the cell, including its bounding box, label, and score.
        class_to_visualize (str): The class label of cells to visualize.
        cropped_table (PIL.Image): The cropped table image.

    Raises:
        ValueError: If the provided class_to_visualize is not one of the available classes.

    Returns:
        None
    """
    if (
        class_to_visualize
        not in TABLE_STRUCTURE_RECOGNITION_MODEL.config.id2label.values()
    ):
        raise ValueError("Class should be one of the available classes")

    plt.figure(figsize=(16, 10))
    plt.imshow(cropped_table)
    ax = plt.gca()

    for cell in cells:
        score = cell["score"]
        bbox = cell["bbox"]
        label = cell["label"]

        if label == class_to_visualize:
            xmin, ymin, xmax, ymax = tuple(bbox)

            ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    color="red",
                    linewidth=3,
                )
            )
            text = f'{cell["label"]}: {score:0.2f}'
            ax.text(
                xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5)
            )
            plt.axis("off")


def get_cell_coordinates_by_row(table_data):
    """
    Generates cell coordinates for each row in a table based on the provided table data.

    Args:
        table_data (list): A list of dictionaries representing objects detected in the table.
            Each dictionary contains information about the detected object, including its
            label and bounding box.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the table
        and contains information
            about the cell coordinates within that row. Each dictionary has the following keys:
                'row': The bounding box of the row.
                'cells': A list of dictionaries representing the cell coordinates within the row.
                Each dictionary has the following keys:
                    'column': The bounding box of the column associated with the cell.
                    'cell': The bounding box of the cell itself.
                'cell_count': The number of cells in the row.
    """
    # Extract rows and columns
    rows = [entry for entry in table_data if entry["label"] == "table row"]
    columns = [entry for entry in table_data if entry["label"] == "table column"]

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x["bbox"][1])
    columns.sort(key=lambda x: x["bbox"][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [
            column["bbox"][0],
            row["bbox"][1],
            column["bbox"][2],
            row["bbox"][3],
        ]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({"column": column["bbox"], "cell": cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x["column"][0])

        # Append row information to cell_coordinates
        cell_coordinates.append(
            {"row": row["bbox"], "cells": row_cells, "cell_count": len(row_cells)}
        )

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x["row"][1])

    return cell_coordinates


def apply_ocr(cropped_table, cell_coordinates, min_accuracy):
    """
    Applies optical character recognition (OCR) to a cropped table
    image and returns the text data as a dictionary.

    Args:
        cropped_table (PIL.Image): The cropped table image to be processed.
        cell_coordinates (list): A list of dictionaries, each containing the
        coordinates of the cells in a row of the table.
        min_accuracy (float): The minimum confidence score for the OCR result to be accepted.

    Returns:
        dict: A dictionary of lists, where each key is the row index and each value is a list of strings representing the cell text in that row.

    Raises:
        ValueError: If the cropped_table is not a PIL.Image object or the cell_coordinates is not a list of dictionaries.
        RuntimeError: If the OCR_READER fails to read the cell images.
    """
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(cell_coordinates):
        row_text = []
        for cell in row["cells"]:
            # crop cell out of image
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            # apply OCR
            result = OCR_READER.readtext(np.array(cell_image))
            if len(result) > 0:
                text = " ".join([x[1] for x in result if x[2] > min_accuracy])
                row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data


def pdf_to_jpg(pdf, output_dir, image_format="jpg"):
    """Converts a PDF file to JPG images, handling errors and creating necessary directories.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Path to the output directory for JPG images.
        image_format (str, optional): Desired image format (default: "jpg").

    Returns:
        list: List of paths to the generated JPG files.
    """

    images = convert_from_bytes(pdf)
    paths = []
    for i, image in enumerate(images):
        path = output_dir + f"image{i}.jpeg"
        paths.append(path)
        image.save(path, "JPEG")

    return paths


def open_image_with_path(image_path):
    """
    Opens an image file from a given path and converts it to RGB format.

    Args:
        image_path (str): The path to the image file.

    Returns:
        PIL.Image: The opened and converted image, or None if the file
                   could not be opened.

    """
    return Image.open(image_path).convert("RGB")


def open_image_in_memory(image):
    """
    Opens an image from a byte array in memory and converts it to RGB format.

    Args:
        image (bytes): The image data as a byte array.

    Returns:
        PIL.Image: The opened and converted image, or None if the image data
                   is invalid.

    """

    image = image.read()
    image_file = io.BytesIO(image)
    return Image.open(image_file).convert("RGB")
