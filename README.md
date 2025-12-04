# Patent Dashboard Preprocessed Datasets

This repository contains preprocessed datasets for the Patent Dashboard application.

## Files

- **preprocessed_sample.csv** (15MB) - Main dashboard dataset with 50,000 sampled patents
- **preprocessed_ai_sample.csv** (5.8MB) - AI patents sample with 18,485 patents
- **cleaned_patents_academic.csv** (8.0MB) - Academic patents dataset with 24,289 patents

## Usage

These datasets are used by the Patent Dashboard deployed on Google Cloud Run. The application downloads them from raw GitHub URLs to reduce Docker image size.
