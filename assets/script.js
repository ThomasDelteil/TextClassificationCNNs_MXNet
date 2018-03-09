var myApp = angular.module('myApp', []);
myApp.controller('AppController', ['$scope', '$http', '$httpParamSerializerJQLike', function ($scope, $http, $httpParamSerializerJQLike) {

    $scope.text = "Cloud Atlas is a very interesting piece. I really like the adaptation, it respects the format and the atmosphere of the book.";

    $scope.catModels = ['crepe'];
    
    $scope.categories = {
            'Books': {
                    'title': 'Books',
                    'url':  'assets/img/books.png',
                    'score': {'crepe':0}
            },
            'CDs_and_Vinyl': {
                    'title': 'CDs & Vinyl',
                    'url':  'assets/img/cd-record.png',
                    'score': {'crepe':0}
            },
            'Home_and_Kitchen': {
                    'title': 'Home & Kitchen',
                    'url':  'assets/img/kitchen.png',
                    'score': {'crepe':0}
            },
            'Clothing_Shoes_and_Jewelry': {
                    'title': 'Clothing, Shoes & Jewelry',
                    'url':  'assets/img/clothes.png',
                    'score': {'crepe':0}
            },
            'Movies_and_TV': {
                    'title': 'Movies & TV',
                    'url':  'assets/img/video-camera.png',
                    'score': {'crepe':0}
            },
            'Cell_Phones_and_Accessories': {
                    'title': 'Phones & Accessories',
                    'url':  'assets/img/smartphone.png',
                    'score': {'crepe':0}
            },
            'Sports_and_Outdoors': {
                    'title': 'Sports & Outdoors',
                    'url':  'assets/img/american-football.png',
                    'score': {'crepe':0}
            }
    }
        

    var scoreCategories = function (models) {
        for (var i = 0; i < models.length; i++) {
            (function (model) {
                $http({
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    url: 'http://localhost:8081/CREPE/predict',
                    data: $httpParamSerializerJQLike({data: JSON.stringify([$scope.text])})
            }).then(function successCallback(response) {
                var scores = response.data.prediction.confidence;
                for (var key in scores) {
                    if (!scores.hasOwnProperty(key)) continue;
                        console.log(key);
                        $scope.categories[key].score[model] = scores[key]*100;
                    
                }
                console.log($scope.categories);
            }, function errorCallback(response) {
                console.log("There was an error with loading the data sources");
            });
            })(models[i]);
        }
    }
        
    // call the scoring API
    var timeout = null;
    var cancel = 0;
    $scope.rescoreData = function() {
        if (timeout != null) {
            if (cancel < 15) {
                cancel += 1;
                clearTimeout(timeout);
            } else {
                cancel = 0;
            }
        }
        timeout = setTimeout(function() {
            scoreCategories(['crepe']);
            cancel = 0
        }, 300);
    }

    $scope.rescoreData();

}]);