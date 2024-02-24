import os
import pandas as pd

from django.shortcuts import render
from django.views import View
from django.http import HttpResponse, FileResponse


from table_extractor.utils import pdf_to_jpg, open_image_in_memory, open_image_with_path
from table_extractor.controller import (
    get_table_from_image,
    get_table_cells_coordinates,
    convert_detected_table_to_csv,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)


class TableExtractor(View):
    template_name = "table_extractor/table_extractor.html"
    min_output_accuracy = 0.4

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        file = request.FILES["file"]

        if file.name.endswith(".pdf"):
            images = pdf_to_jpg(file.read(), os.path.join(BASE_DIR, "files"))

            # TODO append data of all the images in the CSV, right now only first
            # file is being converted to csv
            image_file = open_image_with_path(images[0])

        else:
            image_file = open_image_in_memory(file)

        cropped_table = get_table_from_image(image_file)
        table_cell_coordinates = get_table_cells_coordinates(cropped_table)
        csv = convert_detected_table_to_csv(
            cropped_table,
            table_cell_coordinates,
            os.path.join(BASE_DIR, "files"),
            self.min_output_accuracy,
        )

        df = pd.read_csv(csv)
        df = df.fillna("")
        context = {
            "table_data": df.values.tolist(),
            "headers": list(df.columns),
        }

        return render(request, self.template_name, context)
    
def download_csv(request):
    csv_file = os.path.join( BASE_DIR, 'files', "output.csv")

    if os.path.exists(csv_file):
        csv_file = open(csv_file, "rb")
        response = FileResponse(
            csv_file, content_type="text/csv", filename="output.csv"
        )
        response["Content-Disposition"] = (
            f'attachment; filename="output.csv"'
        )
        return response
    else:
        return HttpResponse("File not found")
