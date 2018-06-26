# Deploying the model using [MXNet Model Server](https://github.com/awslabs/mxnet-model-server)

Instruction to install and more details [here](https://github.com/awslabs/mxnet-model-server).
We are essentially starting a flask server that will be able to handle reviews and return the best match among the 7 categories that we trained on.

## Model creation

To create a new `.model` file run:
```
mxnet-model-export --model-name crepe_cnn --model-path . --service-file-path crepe_cnn_service.py
```

## Testing locally

To start the server run:
```
mxnet-model-server --models CREPE=crepe_cnn.model
```

To test the API run

```
curl -X POST http://127.0.0.1:8080/CREPE/predict -F "data=['I love this book']"
```

You should get back:
```
{
  "prediction": {
    "confidence": {
      "Books": 0.986,
      "CDs_and_Vinyl": 0.0,
      "Cell_Phones_and_Accessories": 0.0,
      "Clothing_Shoes_and_Jewelry": 0.0,
      "Home_and_Kitchen": 0.003,
      "Movies_and_TV": 0.005,
      "Sports_and_Outdoors": 0.002
    },
    "predicted": "Books"
  }
}
```

## Serving the model using Docker

Some helpers commands:

- To build the docker image run:

`make build`

- To run the docker container (server on port 8081) run:

`make run`

The server will automatically start inside the container. Similarly you can call it by sending a `curl` request:

```
> curl -X POST http://127.0.0.1:8081/CREPE/predict -F "data=['I recommend this book Machine Learning from Tom Mitchell']"
{
  "prediction": {
    "confidence": {
      "Books": 0.992,
      "CDs_and_Vinyl": 0.0,
      "Cell_Phones_and_Accessories": 0.0,
      "Clothing_Shoes_and_Jewelry": 0.0,
      "Home_and_Kitchen": 0.001,
      "Movies_and_TV": 0.002,
      "Sports_and_Outdoors": 0.002
    },
    "predicted": "Books"
  }
}
```

- To monitor the log

`make monitor`
