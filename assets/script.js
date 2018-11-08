var myApp = angular.module('myApp', []);
myApp.controller('AppController', ['$scope', '$http', '$httpParamSerializerJQLike', function ($scope, $http, $httpParamSerializerJQLike) {

    $scope.text = "Dr. Bishop is one of the brightest minds working in the field of pattern recognition and machine learning. This book, appropriate for study at the advanced undergraduate level, discusses the major topics and techniques.";

    $scope.catModels = ['crepe'];
    
    $scope.categories = {
            'Books': {
                    'title': 'Books',
                    'url':  'assets/img/books.png',
                    'score': 50
            },
            'CDs_and_Vinyl': {
                    'title': 'CDs & Vinyl',
                    'url':  'assets/img/cd-record.png',
                    'score': 40
            },
            'Home_and_Kitchen': {
                    'title': 'Home & Kitchen',
                    'url':  'assets/img/kitchen.png',
                    'score': 30
            },
            'Clothing_Shoes_and_Jewelry': {
                    'title': 'Clothing, Shoes & Jewelry',
                    'url':  'assets/img/clothes.png',
                    'score': 20
            },
            'Movies_and_TV': {
                    'title': 'Movies & TV',
                    'url':  'assets/img/video-camera.png',
                    'score': 50
            },
            'Cell_Phones_and_Accessories': {
                    'title': 'Phones & Accessories',
                    'url':  'assets/img/smartphone.png',
                    'score': 50
            },
            'Sports_and_Outdoors': {
                    'title': 'Sports & Outdoors',
                    'url':  'assets/img/american-football.png',
                    'score': 60
            }
    }
        

    var scoreCategories = function () {
        if ($scope.text.length < 3) {
            $scope.predicted = "";
            // case where there is nothing we want to show 0 everywhere

            for (var key in $scope.categories) {
                if (!$scope.categories.hasOwnProperty(key)) continue;
                    $scope.categories[key].score = 0;
            }
            $scope.$apply();
        } else {
            $http({
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'},
                //url: 'https://api.thomasdelteil.com/predictions/crepe',
                //data: $httpParamSerializerJQLike({"data": JSON.stringify([{"review":$scope.text,"review_title":""}])})
                url: 'http://localhost:8081/CREPE/predict',
                data: $httpParamSerializerJQLike({data: JSON.stringify([$scope.text])})
            }).then(function successCallback(response) {
                var scores = response.data.prediction.confidence;
                $scope.predicted = response.data.prediction.predicted;
                for (var key in scores) {
                    if (!scores.hasOwnProperty(key)) continue;
                    $scope.categories[key].score = scores[key] * 100;
                }
            }, function errorCallback(response) {
                console.log("There was an error with loading the data sources");
            });
        }
    };
        
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
            scoreCategories();
            cancel = 0
        }, 300);
    }

    $scope.randomReview = function() {
        $scope.text = $scope.reviews[Math.floor(Math.random()*$scope.reviews.length)];
        $scope.rescoreData();
    }

    $scope.rescoreData();

    $scope.reviews = [
        "Lovin' these! | I absolutely love these resistance bands! You have every amount of resistance you could need, and bands can be easily added together to get your perfect resistance. The handles are comfortable, and the door anchor does stay in the door. I was a little scared to try the door anchor because I was afraid it was going to come flying out and hit me in the face, but I tested it using a long broom lol. ",
        "Nexus 5 | Excellent fit with all the cut-outs perfectly placed... on-off/volume control work flawlessly.... case shows off the N-5 with minimal bulk and all with a reasonable cost!!!!",
        "Great knife | This big fella has a really nice feel to it and is really helpful for stripping bark off hiking sticks.  Very sturdy.",
        "Sympathy for the people | This is a very strange album , it's actually not really an LP , it's kinda like the extra from Portrait Of An American Family . Smells like children is awsome to listen to in the dark , and the Dope Hat remixes are good . The Album is a big joke , and it's hilarious .",
        "My favorite jeans | Carhartt jeans are the best in my book. Been wearing this cut for years and probably will for many more. reasonably priced jeans that wear well. I'm a cheapskate at heart...these are right up my alley!",
        "Excellent cycle gloves | Great cycling gloves.  Affordable and very comfortable on long rides.  I use these both rode and mountain bike riding.  I highly recommend these.",
        "Cast Iron and SuperSaver Shipping = Woo Hoo! | I have had a couple of cast iron skillets prior to purchasing this Dutch Oven from Amazon. This Oven is great. I use it for stews and beans and sourdough bread and all sorts of things.Because I will use this in the oven rather than over a campfire, the  handles on the side of the Dutch Oven work much better that the wire bail handle.",
        "Canadian Thriller | A some what decent cold war spy thriller from Canada, loosely based on real life characters. A U.S. Naval officer turned CIA spy to catch the infamous \"Carlos the Jackal\". Ben Kingly and Donald Sutherland are the veteran actors in this film",
        "Amazing Grace | Fantastic book!  I was surprised how little I knew about the history of the slave trade.  The depravity that allowed such an abomination to continue so long for profit is simply shocking. We tend to think of slavery as exclusive to the American south.  In fact, Europe shares the burden of guilt for this horror",
        "Great fit | Initially, I would've given this 5 stars, however, after 2-ish years, it has spots of rust on it.  I'm not happy with that, however, the way the lid opens easily with my elbow, foot, or whatever's available, the shape of this can (round doesn't fit in the space available), and the fact that it has an interior plastic can, I would still buy it again despite the rust.",
        "Gets the job done | I do really like this mixer. It is my first stand mixer. I really really want a Kitchenaid but due to price I had to settle for this one for now. I wish that the big bowl was able to be even bigger",
        "The Real Jesus by Luke Timothy Johnson | The BIBLE is the inspired WORD of GOD.  Therefore, what God wants his creation to know is written there and interpreted by the Holy Spirit.  Those who are trying to 'dig up' new information are trying to draw attention to self & away from the WORD.",
        "RUSH is at the top of their game with this masterpiece | Moving Pictures was the first RUSH cd that I ever purchased. It's tied for #1 as my favorite with Permanent Waves and 2112. The one drawback about this remastered cd, is that when I play this remaster with my original issue, the remastered cd sounds about the same",
        "Does what other molds can't! | If you are looking for a mold that works this one is the answer you have been searching for. Plastic molds I have used in the past stick and eventually break",
        "Very Nice | This is a very nice jacket.  I wear it with black pants that are made of the same material so it looks like I have on a business pants suit on.",
        "comfortable | This is more like a shoe than sneaker, offering room for orthotics and comfort. It has a very roomy toe box and stability also.",
        "One of the greatest debut albums of all time. | EVERY SINGLE SONG ON THIS ALBUM IS GREAT.  A MUST OWN FOR CLASSIC ROCK/'70s POP FANS.",
        "How To Marry a Millionaire Vampire | I am currently in the middle of this book but so far I love it.  I love the humor and the characters.  There is nothing I love better than a book that can make me laugh out loud and this one does.",
        "Junk JuNk JUNK | Belt clip will break! Case is alright but the plastic is brittle. WASTE of money if you ask me but you get what you pay for. *shrugs*",
        "front glass lcd display | I love it. It fits perfectly. Well worth the $20 spend. If I need to do it again I sure will.",
        "No Grip | This case was what I expected in regards to protection, however the grip was not there. Had this case had maybe a rubber grip on both sides... This case would be perfect.",
        "Perfect Costume Component | These glasses gave me just the nerd look I was looking for when I dressed up as Steve Urkel. After a while, I forgot that I had them on because they fit so well.",
        "Roars to life! | You should know that I absolutely adore James Cagney and will consider him one of the greatest actors of any generation for as long as I am alive.  The man was spectacular, and while he certainly had his niche, he brought too much to it over his fantastic career."
    ]
}]);
