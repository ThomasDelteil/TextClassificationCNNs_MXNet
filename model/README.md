# Deploying the model using [MXNet Model Server](https://github.com/awslabs/mxnet-model-server)

Instruction to install and more details [here](https://github.com/awslabs/mxnet-model-server).
We are essentially starting a flask server that will be able to handle reviews and return the best match among the 7 categories that we trained on.

To create a new `.model` file run:
```
mxnet-model-export --model-name crepe_cnn --model-path . --service-file-path crepe_cnn_service.py
```

To start the server run:
```
mxnet-model-server --models CREPE=crepe_cnn.model --service "crepe_cnn_service.py"
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
