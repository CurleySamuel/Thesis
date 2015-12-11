# Realtor in a Regressor: Predicting Boston Area Home Sale Prices Using Only Physical Features

# Getting Zillow-like Performance with a Subset of the Data: Predicting Boston Area Home Sale Prices Using Only Physical Features

# A Hard Look at the Soft Science of Real Estate: Predicting Boston Area Home Sale Prices Using Only Physical Features

# Abstract

Predicting the sale price of a home is easy when given the entire history of the property including past sale prices, tax assessments, and physical attributes. Take the localized market growth, apply it to the last sale price and tack on any additions or subtractions to credit new features. However predicting the sale price of a home when given only physical attributes is a much harder problem. The regression models are no longer seeded with a known historical baseline - they instead have to look a layer deeper and start attributing features with values, location coordinates with land value, and realtor comments with discrete dollar deltas.

The goal isn't to beat common home valuation tools like Zillow and Redfin at their own game. Instead the goal is to get similar accuracy using a much smaller subset of data for each home, not to mention a much smaller amount of training data. The smaller dependence on data opens more doors as there's no longer a strict dependence on having historical data and tax information. Using only twenty-five features which can easily be specified by a user on a phone we can generate a valuation that's almost as good as one based on myriad data sources and historical data that may or may not be emotionally influenced.

# Table of Contents

- [Introduction](#Introduction)
- [Methods](#Methods)
  - [The Data](#The-Data)
  - [Iteration 0](#Iteration-0)
  - [Iteration 1](#Iteration-1)
  - [Iteration 2](#Iteration-2)
  - [Iteration 3](#Iteration-3)
  - [The Models](#The-Models)
- [Results](#Results)
  - [Scoring](#Scoring)
  - [Iteration 0](#Iteration-0)
  - [Iteration 1](#Iteration-1)
  - [Iteration 2](#Iteration-2)
  - [Iteration 3](#Iteration-3)
- [Discussion](#Discussion)
- [Conclusions](#Conclusions)
- [Acknowledgements](#Acknowledgements)
- [References](#References)



# Introduction

A decade ago the only way to get a trustworthy valuation of your home was to call in a realtor to do a comparative market analysis (CMA). Unfortunately doing a CMA is a lengthy process that can take anywhere from 2 - 8 hours and as such either payment or the promise of business for that realtor is a typical prerequisite. The curious prospective seller wanting to get an idea of the value of their home was limited to the crude methods of scoping out nearby open houses and word of mouth exchanges.

But with the internet revolution came a revolution in home valuation - tools like Zillow's Zestimate started popping up promising to give you an instant valuation of your home from your browser. No realtors, no coy smalltalk with neighbors, no commitment to selling your home - the curiosity of the prospective seller can now be satiated. But just how accurate are these new predictions and what data do they use?

Nationally Zillow manages a median error rate of 7.9% which drops to 7.5% when only considering the Boston area. Of those 1.5 million Bostonian homes - 35.3% of the Zestimates are within 5% of the actual sale price; 62.6% within 10%; 87.8% within 20%; leaving 12.2% that are more than 20% off.


Some of the data that Zillow uses to achieve these estimates are -

  - Physical attributes
    - Location
    - Square footage
    - Number of bedrooms, bathrooms.
    - Etc.
  - Tax assessments
    - Property tax information
    - Actual property taxes paid
    - Etc.
  - Prior history
    - Actual sale prices of the home
    - Comparable recent sales of nearby homes


Another home valuation service Redfin claims to use 500 data points ranging from the neighborhood the home's in to specialized qualitative information like whether the home has a water view or the calculated noise pollution levels. That's all well and good but these models have a high dependence on the availability of this information and they start breaking down when considering brand new homes that have no history or no tax records. That's the scenario that we're going to target. In fact the explicit list of features that we'll consider is below -

  - Address
  - Zip-code
  - Age
  - Square footage
  - Lot size
  - Style
  - Number of bedrooms
  - Number of bathrooms
  - Number of garages
  - Number of floors
  - Number of fireplaces
  - Type of roofing
  - Type of flooring
  - Type of appliances
  - Type of foundation
  - Type of construction
  - Type of exterior
  - Type of insulation
  - Type of water heater
  - Type of heating
  - Type of wiring
  - Exterior features
  - Interior features
  - Basement
  - Realtor remarks

We'll later find that most of these are relatively unimportant features and can be removed without much effect on the accuracy of the model. Regardless they're all specified whenever a home is entered into MLS (multiple listing service) and thus serve a good starting point as every home will have at least these features. It's also a good starting point because it's all information that a home owner should be able to come up with fairly easily if given drop downs of options for each. If made accurate this would open up the world of CMA's and appraisals as they no longer require specialized knowledge but can be performed by the home owner with only a phone.

But the path to accuracy won't be easy. We'll take these 25 features and transform them into over 322 features by geolocating addresses, clustering homes in several dimensions, analyzing the paragraphs of remarks left by realtors, and tuning the hyper parameters of chained models. From that incredible amounts of data we'll generate a single number for every home - our predicted sale price of the home if it was sold today. May the odds be ever in our favor.


# Methods

The project was done with an iterative development model in mind and as such it only makes sense to present the project using it's iterations. The first iterations were primarily dedicated to building out the data pipeline and normalization techniques but also featured crude models based on small subsets of the data. Then the models started getting more refined as more and more features were incorporated into the models via more feature extraction and text analysis. The final iterations had the smallest code delta but took the most amount of time as they were heavy on tuning model hyper-parameters and soaking as much accuracy out from them as possible.


## The Data

The data for 21,657 homes were sourced from the multiple listing service (MLS).

## Iteration 0

Location, location, location. Unfortunately _221B Baker Street_ doesn't give our models much to work with so the first preprocessing step is to geocode our entire suite of testing and training data. The fundamental flaw with an address is that addresses on their own don't contain any proximity information beyond the street name - are _17 Welthfield St_ and _2998 Homer Ave_ close to each other (and thus have similar land value)? It's impossible to say without first converting the address to their geographical coordinates. Suddenly the problem of proximity is reduced to euclidean distance. Geocoding in itself isn't a mystery so I'll skim over the implementation details but the source code can be found in [geocode_data.py](geocode_data.py).

Also in this iteration was the first foray into regressors. The first iteration ended with six distinct regression models with each model using either a different regression algorithm or a completely different subset of the data. Specifically we have -

- Linear Regressor using only the square footage of a home
- Random Forest Regressor using only the location of the home
- Random Forest Regressor using only `LAT/LNG/AGE/SQFT/BEDS/BATHS/GARAGE/LOTSIZE`
- Random Forest Regressor using all but realtor remarks
- Extremely Randomized Trees Regressor using all but realtor remarks
- Gradient Boosted Regression Trees using all but realtor remarks

Worth noting is the heavy reliance on ensemble methods here. An ensemble method (i.e. RFR, ERTR, GBRT above) is an estimator constructed from either averaging methods that average the results of several simpler estimators or boosting methods which construct successive generations of simple estimators each based on the error rate of the previous. The reliance on ensemble methods is because they were quite simply the empirical best. I tried a variety of generalized linear models, support vector machines and decision trees but kept coming back to the ensemble methods.

There's also the small detail I skipped over regarding how I'm incorporating the non-numerical data into my models. Take the 'Hot Water' feature as an example - it's possible values are any combination of the set of options `Electric, Geothermal/GSHP Hot Water, Leased Heater, Natural Gas, Oil, Propane Gas, Separate Booster, Solar, Tank, Tankless, Other`. To handle that I take all categorical features and generate several indicator features, one for every unique category in the feature. The 'Hot Water' feature is thus split into eleven distinct features each containing a binary number for whether or not that specific category within 'Hot Water' is present. Specifically we generate the columns `(Hot Water) electric, (Hot Water) geothermal/gshp hot water, (Hot Water) leased heater, (Hot Water) natural gas, (Hot Water) oil, (Hot Water) other (see remarks), (Hot Water) propane gas, u(Hot Water) separate booster, (Hot Water) solar, (Hot Water) tank, (Hot Water) tankless`.

The unfortunate consequence of this feature extraction is that we start with matrix dimensions `(21657, 20)` and end with matrix dimensions `(21657, 323)`. As we find out later this can almost end up hurting us as highly correlated and valuable features can get lost in the pool of relatively insignificant features.

## Iteration 1

In this iteration we took the best three models from the previous iteration and introduced cross validation, a technique used to minimize overfitting (see [Scoring](#Scoring)). The selected models are below -

- Random Forest Regressor using all but realtor remarks
- Extremely Randomized Trees Regressor using all but realtor remarks
- Gradient Boosted Regression Trees using all but realtor remarks

We then picked the best performing model of the three after cross validation and starting tuning the model's hyperparameters. A hyperparameter of a model are more or less a specific set of constants used in a model that affect the ultimate accuracy of the model. In the case of gradient boosted regression trees the hyperparameters include (list and descriptions courtesy of scikit-learn documentation) -


- ###### loss : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}, optional (default=’ls’)

 loss function to be optimized. ‘ls’ refers to least squares regression. ‘lad’ (least absolute deviation) is a highly robust loss function solely based on order information of the input variables. ‘huber’ is a combination of the two. ‘quantile’ allows quantile regression (use alpha to specify the quantile).
- ###### learning_rate : float, optional (default=0.1)

 learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
- ###### n_estimators : int (default=100)

 The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
- ###### max_depth : integer, optional (default=3)

 maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree.
- ###### min_samples_split : integer, optional (default=2)

 The minimum number of samples required to split an internal node.
- ###### min_samples_leaf : integer, optional (default=1)

 The minimum number of samples required to be at a leaf node.
- ###### min_weight_fraction_leaf : float, optional (default=0.)

 The minimum weighted fraction of the input samples required to be at a leaf node.
- ###### subsample : float, optional (default=1.0)

 The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter - n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
- ###### max_features : int, float, string or None, optional (default=None)

 The number of features to consider when looking for the best split:
- ###### alpha : float (default=0.9)

 The alpha-quantile of the huber loss function and the quantile loss function.


While it may seem persnickety to worry about fine-tuning models already, the term 'fine-tune' is perhaps a misnomer. While fine-tuning my models I noticed some configurations which achieved scores significantly above the default but I also achieved scores that were in the single digits. Just by lowering the minimum weighted fraction of samples required to be a leaf I could end up reducing my model accuracy from 0.88 to 0.02. It also means that perhaps the default model isn't performing as well as it can and we may be able to squeeze out a significant bit of accuracy before moving on to more complex additions. There are three primary techniques for tuning the hyperparameters of a model. You can guess, you can try random combinations, or you can try every possible combination. I tried all three.

### Guessing

There's one variable which guessing works quite well with - n_estimators which is the number of boosting stages to perform. Gradient Tree Boosting is an ensemble method in that it forms ensembles of weaker decision learners, each ensemble improving on their predecessor's estimate by chaining a new tree on the error of the last tree. The number of boosting stages is how many of these improvement stages occur. We can guess this constant because at every iteration if we measure the deviance of both the training set and the testing set and graph this deviance, we can tell when improvement starts slowing down. From there we can select a reasonable trade off between training time and trailing off accuracy improvements.

![Graph of Deviance vs. Iterations](/images/figure_1.png)

![Graph of Improvement vs. Iterations](/images/figure_2.png)

### Randomly Guessing

The Python machine learning framework Scikit-learn (colloquially known as 'sklearn') includes a helper module with utilities to fine-tune models. One of these utilities is RandomSearch. In RandomSearch you define distributions or lists of possible values for each parameter. RandomSearch will then run for N iterations, at each iteration randomly selected each parameter, fitting a model with those parameters and scoring the fitted model. After hitting the specified number of iterations it'll then report the best parameter combination it's had so far. While not exhaustive it tends to perform quite well while trying significantly fewer combinations than the next utility, GridSearch.

### Guessing Everything

Another one of those utilities in sklearn is GridSearch. How GridSearch works is you define a list of possible values for every parameter defining a grid of possible combinations. GridSearch will then iterate over every combination, train your model using that combination of parameters and score it's accuracy. After exhausting the space of combinations it'll then report the top N configurations that produced the highest score. While incredibly useful given a small set of combinations, because of innate combination theory this utility is often time prohibitive when you want to search a large grid of combinations.


## Iteration 2

## Iteration 3

## The Models

We've talked about a variety of models and algorithms used to form our estimation pipelines. While this paper does expect some level of statistical knowledge; it doesn't make the leap that the reader is familiar with the intricacies of various regression models. As such I wanted to include a brief section dedicated to a quick overview of the models and algorithms that we've seen. Because my primary machine learning library used was scikit-learn - all regressors and algorithms below used the implementation that can be found in that package.

### Regression

- #### Random Forests

 A random forest is a meta-estimator formed by an ensemble of decision trees each fitted over different subsamples of the dataset. The output value is then the mean of each decision tree's output values. Random forests are versatile and often the goto model when facing a new problem as they can help uncover the relative importance of features.

- #### Extremely Randomized Forests

 Functionally equivalent with random forests but introduce more randomness when selecting a subset of features. This tends to reduce the variance of the model in exchange for a slight increase in bias.

- #### Gradient Tree Boosting  

 Another ensemble method, gradient boosted regression trees form ensembles of weaker decision learners that improve on their predecessor's estimate by chaining a new tree on the error of the last tree.

- #### Least Angle

 Least angle regression is a linear regression algorithm typically used when overfitting is a concern or when analyzing a sparse matrix.

### Clustering

- #### MeanShift

 Mean shift by itself is a technique for locating maximas in a density function. Mean shift clustering works by assuming that the input data is a sampling from an underlying density function. It then uses mean shift appropriately to uncover modes in the feature space (clusters of similar values).

### Matrix Decomposition

- #### Truncated Singular Value Decomposition

 Matrix decomposition algorithm used to reduce the dimensionality of an input matrix. In the context that we're using it (word frequency matrices) it's actually called latent semantic analysis.

### Feature Extraction

- #### tf-idf

 Term frequency inverse document frequency. This term frequency transformation will reweight the term frequency of a blob by the inverse of the commonality of the term across all blobs. That is, it'll scale down the reported frequencies of common words and scale up the frequencies of unique words. This is a common technique to reflect word importance and help summarize text.


# Results

## Scoring

We need a method of scoring the accuracy of our regression models and one of the primary ways we're going to do so is the coefficient of determination (R<sup>2</sup>). R<sup>2</sup> has the range [0.0, 1.0] and represents the fraction of variance in the output that's predictable from the input. While generally a good scoring method it does suffer from a few problems like when sample size increases as R<sup>2</sup> rewards larger sample sizes (fixed by adjusted R<sup>2</sup>). An R<sup>2</sup> of 1.0 means that the model is predicting perfectly.

To allow us to compare our regression capabilities to that of Zillow we'll also be looking at the mean absolute percent error in a few cases. MAPE on it's own isn't a very good scoring method as predictions that are systematically low are bounded to a MAPE of 100% while predictions that are systematically too high are unbounded. This means that when using MAPE to compare models there exists a bias towards estimators that predict lower versus models that predict higher. A MAPE of 0.0 means the model is predicting perfectly.

We also face the issue of overfitting. The usual method of dealing with overfitting is handled by segmenting your data into two distinct sets - a training set and a testing set. But this still isn't perfect because when tuning the hyperparameters of models we'll favor the parameters that perform the best on the testing set (but not necessarily new data). To handle this overfitting problem we introduce the idea of cross validation, specifically K-Fold cross validation. K-Fold CV segments the dataset into K distinct but equal subsamples. K-1 of the subsamples are used to train the model and the remaining subsample is then used as the testing data. This is repeated K times so that every subsample has the chance to be the testing set. The resultant score is usually the average of the K subscores.


## Iteration 0

For every model in this iteration we've included both the R<sup>2</sup> score as well as a sample of predictions made. Next to our predicted values are both the list price and the sold price of the respective home which allows us to gauge what an 'acceptable' value is.

##### Linear Regressor using only the square footage of a home
```
          Accuracy: 0.459411179095

          Predicted                    List Price                     Sale Price         
           537,373                        289,500                        295,000          
           788,038                        850,000                        820,000          
           563,946                        449,900                        430,000          
           647,936                        699,900                        660,000          
           588,858                        749,000                        805,000          
           658,257                        585,000                        570,000          
           724,452                        719,900                        708,000          
           677,000                        679,000                        700,000          
           581,266                        465,000                        450,000          
           833,355                       1,058,000                      1,000,000         
           477,465                        349,900                        360,000          
           489,328                        315,000                        295,000          
           536,780                        379,900                        370,500          
           616,262                        549,900                        547,000          
           478,414                        279,900                        273,000          
           532,509                        349,000                        337,000          
           587,079                        443,900                        420,000          
           573,911                        449,900                        455,000          
           680,559                        599,900                        575,000          
           515,426                        339,900                        339,900          
```

##### Random Forest Regressor using only the location of the home
```
          Accuracy: 0.512581201614

          Predicted                   List Price                     Sale Price          
          392,130                        779,000                        816,250           
          463,825                        475,000                        450,000           
          569,080                        589,900                        587,400           
          491,710                        975,000                        950,000           
          286,170                        389,000                        395,000           
          632,695                        319,000                        319,000           
          360,395                        419,900                        408,000           
          610,860                        425,000                        414,000           
          419,760                        519,900                        499,900           
          415,287                        459,900                        460,000           
          445,330                        445,000                        420,000           
          1,437,681                       549,000                        522,000          
          280,043                        249,900                        251,000           
          455,712                        459,900                        420,000
          547,539                        699,800                        665,000           
          867,808                        819,900                        824,723           
          363,740                        339,900                        323,000           
          395,186                        575,000                        555,000           
          921,068                       1,449,000                      1,485,000          
          1,067,873                      1,325,000                      1,210,000
```

##### Random Forest Regressor using only `LAT/LNG/AGE/SQFT/BEDS/BATHS/GARAGE/LOTSIZE`

```
          Accuracy: 0.860120147944

          Predicted                   List Price                     Sale Price    
          437,195                        529,900                        540,000           
          671,328                        589,000                        570,000           
          460,552                        359,000                        360,000           
          409,415                        395,000                        372,000           
          466,588                        595,000                        550,000           
          1,556,137                      1,359,000                      1,320,000         
          761,688                        759,000                        725,000           
          321,212                        419,900                        400,000           
          264,517                        274,900                        276,000           
          373,922                        389,900                        389,000           
          449,084                        479,900                        470,000           
          527,548                        475,000                        465,000           
          375,329                        329,900                        315,000           
          627,696                        699,000                        696,000           
          340,283                        349,900                        356,414           
          442,917                        419,900                        407,000           
          197,086                        275,900                        270,000           
          358,586                        389,900                        389,900           
          476,083                        399,900                        399,000           
          462,015                        429,900                        422,000

```

##### Random Forest Regressor using all but realtor remarks
```
          Accuracy: 0.864045570243

          Predicted                   List Price                     Sale Price    
          393,343                        444,900                        420,000           
          650,725                        649,000                        640,000           
          497,629                        489,000                        482,000           
          644,374                        699,000                        678,000
          448,547                        480,000                        472,500           
          441,195                        489,900                        475,000           
          560,107                        585,000                        578,000           
          853,622                        885,000                        990,000           
          575,484                        598,000                        598,000           
          915,815                       1,250,000                      1,195,000          
          837,188                        899,000                        874,500           
          438,348                        539,900                        530,000           
          680,572                        739,900                        705,000           
          416,549                        335,900                        325,000           
          286,449                        199,900                        187,000           
          786,023                        785,000                        782,000           
          419,564                        519,900                        489,000           
          533,769                        499,900                        492,500           
          404,101                        431,900                        415,000
          387,042                        419,900                        420,000
```

##### Extremely Randomized Trees Regressor using all but realtor remarks

```
          Accuracy: 0.882409157472

          Predicted                   List Price                     Sale Price          
          1,250,428                      1,325,000                      1,325,000         
          528,822                        509,900                        485,800           
          577,275                        610,000                        615,000           
          930,412                        959,000                        958,000           
          692,539                        779,000                        764,500           
          425,572                        499,000                        485,000           
          930,020                        949,900                        890,000           
          513,619                        609,000                        595,000           
          503,727                        450,000                        485,000           
          346,979                        299,900                        299,900           
          457,454                        489,900                        485,000           
          619,712                        599,000                        574,000           
          774,598                        899,900                        875,000           
          287,053                        280,000                        263,000           
          732,971                        689,900                        689,000           
          403,004                        459,900                        435,000           
          395,532                        389,900                        370,000           
          740,589                        699,000                        665,000           
          685,062                        629,000                        635,000           
          269,818                        315,000                        300,500           
```

##### Gradient Boosted Regression Trees using all but realtor remarks

```
          Accuracy: 0.885379973123

          Predicted                   List Price                     Sale Price          
          628,326                        649,000                        649,000           
          1,098,072                       949,999                       1,015,000        
          538,834                        595,000                        595,000           
          624,204                        629,900                        597,500           
          650,419                        675,000                        665,000           
          583,247                        575,000                        580,800           
          688,979                        849,000                        815,000           
          434,513                        495,000                        480,000           
          645,526                        729,000                        750,000           
          567,911                        489,000                        481,500           
          577,030                        479,900                        490,000           
          375,504                        389,900                        385,000           
          366,270                        399,000                        375,000           
          493,915                        599,000                        662,000           
          1,417,576                      1,848,000                      1,902,000        
          464,973                        529,000                        523,500           
          320,100                        319,900                        317,500           
          263,166                        279,900                        277,000           
          635,963                        799,900                        765,000           
          953,021                        899,900                        893,000           
```

Judging purely by score the final model using the boosting technique on our entire set of data minus realtor remarks seems to be the most accurate (we'll incorporate the remarks in a later iteration). As you can see in the sample of predictions made there are several cases where the prediction is eerily accurate and gets within $10,000 of the final sale price but there's also a fair share of predictions as far off as half a million dollars. Below is a chart comparing our percentile accuracy of our most accurate model so far with Zillow's reported accuracy in Boston, MA.  

| | Our GBRT | Zillow |
| --------- | :-------: | :----: |
| Within 5% | 0.286 | 0.353 |
| Within 10% | 0.528 | 0.626 |
| Within 20% | 0.823 | 0.878 |

It's getting there. Our numbers aren't quite close enough to serve as a reliable tool yet but it's worth pointing out what a result we're getting with our limited data. Zillow's numbers have access to historical home data including historical sale prices (if a home sold for $760,000 two years ago, predicting it'll sell for $760,000 + relative area growth would be an easy but comparably accurate algorithm), but all our algorithm has is the physical features of a home. It has to play the role of an appraiser using what accounts to only the information you'd find on an real estate shop window flyer, a significantly tougher job. The next iteration will take the best model from this iteration - gradient boosted regression trees, and build upon it.


## Iteration 1

## Iteration 2

## Iteration 3

# Discussion

# Conclusions

# Acknowledgements

# References
