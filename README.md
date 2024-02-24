# Table Extractor - A Python Web Application

## Overview

This repository provides a Python web application that extracts tables from uploaded PDF or image files. It utilizes deep learning models to detect tables and accurately extract their content into a CSV format.

## Features:

Multi-format support: Extracts tables from both PDF and image files.
Deep learning-powered: Leverages pre-trained models for efficient and accurate table detection.
CSV output: Generates a well-formatted CSV file containing the extracted table data.
Error handling: Handles invalid file formats and potential errors gracefully.

## Requirements:

Python 3.x

Django web framework

Necessary libraries (install using pip install -r requirements.txt)

## TODOs and Assumptions:

As UI wasn't the main concerns of the Assignment, I kept it simple.

It takes the first page of the PDF file for table detection, a logic to append all the data should
be applied.

It doesn't use a DB to store and load files from, It stores files internally as of now just for a 
basic demo.

No authentication

The app currently doesn't take it to consideration that many people can visit and request to download the file at the same time.


