# Realtor in a Regressor: Predicting Boston Area Home Sale Prices Using Only Physical Features

# Getting Zillow-like Performance with a Subset of the Data: Predicting Boston Area Home Sale Prices Using Only Physical Features

# A Hard Look at the Soft Science of Real Estate: Predicting Boston Area Home Sale Prices Using Only Physical Features

# Abstract

Predicting the sale price of a home is easy when given the entire history of the property including past sale prices, tax assessments, and physical attributes. Take the localized market growth, apply it to the last sale price and tack on any additions or subtractions to credit new features. However predicting the sale price of a home when given only physical attributes is a much harder problem. The regression models are no longer seeded with a known historical baseline - they instead have to look a layer deeper and start attributing features with values, location coordinates with land value, and realtor comments with discrete dollar deltas.

The goal isn't to beat common home valuation tools like Zillow and Redfin at their own game. Instead the goal is to get similar accuracy using a much smaller subset of data for each home, not to mention a much smaller amount of training data. The smaller dependence on data opens more doors as there's no longer a strict dependence on having historical data and tax information. Using only twenty-five features which can easily be specified by a user on a phone we can generate a valuation that's almost as good as one based on myriad data sources and historical data that may or may not be emotionally influenced.

# Table of Contents

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

## Iteration 1

## Iteration 2

## Iteration 3

# Results

## Iteration 1

## Iteration 2

## Iteration 3

# Discussion

# Conclusions

# Acknowledgements

# References
