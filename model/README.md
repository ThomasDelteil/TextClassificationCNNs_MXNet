## Model Deployment

To deploy your own API, head over to this example in the [MMS example folder](https://github.com/awslabs/mxnet-model-server/tree/master/examples/gluon_character_cnn), you can then build a docker container and deploy it in your favourite cloud. I used [AWS Fargate](https://aws.amazon.com/fargate/).

The static [website](https://thomasdelteil.github.io/TextClassificationCNNs_MXNet/) can be easily deployed straight through github pages, fork the repo and head over to the gh-pages branch to see the html and javascript code.

You will need to update this line https://github.com/ThomasDelteil/TextClassificationCNNs_MXNet/blob/gh-pages/assets/script.js#L63 with your own API urls if you want to use your own classification API.

**Warning** with MMS 1.0.0, CORS support has been lost. However it is already fixed in the master branch, to deploy your own API you will need to use the nightly version of the docker build and specify your CORS setting according to the documentation.
