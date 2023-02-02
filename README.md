# GOBA

GOBA is a tool for the analysis of bacterial genomes. It is designed to be used in a pipeline, and is not intended to be used as a standalone tool. It is written in Python 3 and is compatible with Python 2.7.

## Installation

### Dependencies

GOBA requires the following dependencies:

* [Python 3](https://www.python.org/) (or Python 2.7)
* [FastAPI](https://fastapi.tiangolo.com/)
* [uvicorn](https://www.uvicorn.org/)
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [tensorflow](https://www.tensorflow.org/)
* [scikit-learn](https://scikit-learn.org/stable/)

### Installation

To install GOBA, clone the repository and install the dependencies:

```bash
git clone
cd goba
pip install -r requirements.txt
```

## Usage

### Running the server

To run the server, use the following command:

```bash
uvicorn main:app --reload
```

### Using the API

The API is documented using [Swagger](https://swagger.io/). To access the documentation, go to GET /docs.

<!-- endpoints -->

## Endpoints

### GET /docs

The Swagger documentation for the API.

### POST /Cancer/predict

Predicts the cancer type of a given answers.

### POST /pneumonia/predict

Predicts the pneumonia type of a given photo.

### POST /tuberculosis/predict

Predicts the tuberculosis type of a given photo.

