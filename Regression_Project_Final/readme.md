```python
import pandas as pd
```

# Zillow Project
### Project Goals
- What features drive the housing prices
- Create a model that beats the current baseline

### Project Description
- Construct an ML Regression model that predict propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties.

- Find the key drivers of property value for single family properties. Some questions that come to mind are: Why do some properties have a much higher value than others when they are located so close to each other? Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location? Is having 1 bathroom worse than having 2 bedrooms?

### Data Acquisition & Preparation
- Acuired the Zillow data from prepare.py get_zillow function
- This function acquires cleans and gets it ready for split
- Next is the split function that divides it to train, validate and test


# Exploration
-  According to the features in the zillow dataset, there were atleast 58 features with plenty of information. However, in the end I chose to stick to 6 features. 
-  These features were:
    - bedrooms  - bedroomcnt
    - bathrooms - bathroomcnt
    - squarefeet- calculatedfinishedsquarefeet
    - total_taxes- taxvaluedollarcnt
    - yearbuilt  
    - county     - fips
    
- #### Questions for exploration:

1. #### does the number of bedrooms and bathrooms in a house increase the cost of the home?
2. #### is the price of the house influenced by the county/ location?
3. #### does the age of the house matter when it comes to the price? 
4. #### does the size of the house in sqfeet influence the price?


```python
data_dictionary = pd.read_csv("zillow_data_dictionary.csv")
data_dictionary

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>'airconditioningtypeid'</td>
      <td>Type of cooling system present in the home (i...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'architecturalstyletypeid'</td>
      <td>Architectural style of the home (i.e. ranch, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>'basementsqft'</td>
      <td>Finished living area below or partially below...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>'bathroomcnt'</td>
      <td>Number of bathrooms in home including fractio...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>'bedroomcnt'</td>
      <td>Number of bedrooms in home</td>
    </tr>
    <tr>
      <th>5</th>
      <td>'buildingqualitytypeid'</td>
      <td>Overall assessment of condition of the buildi...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>'buildingclasstypeid'</td>
      <td>The building framing type (steel frame, wood f...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>'calculatedbathnbr'</td>
      <td>Number of bathrooms in home including fractio...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>'decktypeid'</td>
      <td>Type of deck (if any) present on parcel</td>
    </tr>
    <tr>
      <th>9</th>
      <td>'threequarterbathnbr'</td>
      <td>Number of 3/4 bathrooms in house (shower + si...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>'finishedfloor1squarefeet'</td>
      <td>Size of the finished living area on the first...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>'calculatedfinishedsquarefeet'</td>
      <td>Calculated total finished living area of the ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>'finishedsquarefeet6'</td>
      <td>Base unfinished and finished area</td>
    </tr>
    <tr>
      <th>13</th>
      <td>'finishedsquarefeet12'</td>
      <td>Finished living area</td>
    </tr>
    <tr>
      <th>14</th>
      <td>'finishedsquarefeet13'</td>
      <td>Perimeter  living area</td>
    </tr>
    <tr>
      <th>15</th>
      <td>'finishedsquarefeet15'</td>
      <td>Total area</td>
    </tr>
    <tr>
      <th>16</th>
      <td>'finishedsquarefeet50'</td>
      <td>Size of the finished living area on the first...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>'fips'</td>
      <td>Federal Information Processing Standard code ...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>'fireplacecnt'</td>
      <td>Number of fireplaces in a home (if any)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>'fireplaceflag'</td>
      <td>Is a fireplace present in this home</td>
    </tr>
    <tr>
      <th>20</th>
      <td>'fullbathcnt'</td>
      <td>Number of full bathrooms (sink, shower + bath...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>'garagecarcnt'</td>
      <td>Total number of garages on the lot including ...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>'garagetotalsqft'</td>
      <td>Total number of square feet of all garages on...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>'hashottuborspa'</td>
      <td>Does the home have a hot tub or spa</td>
    </tr>
    <tr>
      <th>24</th>
      <td>'heatingorsystemtypeid'</td>
      <td>Type of home heating system</td>
    </tr>
    <tr>
      <th>25</th>
      <td>'latitude'</td>
      <td>Latitude of the middle of the parcel multipli...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>'longitude'</td>
      <td>Longitude of the middle of the parcel multipl...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>'lotsizesquarefeet'</td>
      <td>Area of the lot in square feet</td>
    </tr>
    <tr>
      <th>28</th>
      <td>'numberofstories'</td>
      <td>Number of stories or levels the home has</td>
    </tr>
    <tr>
      <th>29</th>
      <td>'parcelid'</td>
      <td>Unique identifier for parcels (lots)</td>
    </tr>
    <tr>
      <th>30</th>
      <td>'poolcnt'</td>
      <td>Number of pools on the lot (if any)</td>
    </tr>
    <tr>
      <th>31</th>
      <td>'poolsizesum'</td>
      <td>Total square footage of all pools on property</td>
    </tr>
    <tr>
      <th>32</th>
      <td>'pooltypeid10'</td>
      <td>Spa or Hot Tub</td>
    </tr>
    <tr>
      <th>33</th>
      <td>'pooltypeid2'</td>
      <td>Pool with Spa/Hot Tub</td>
    </tr>
    <tr>
      <th>34</th>
      <td>'pooltypeid7'</td>
      <td>Pool without hot tub</td>
    </tr>
    <tr>
      <th>35</th>
      <td>'propertycountylandusecode'</td>
      <td>County land use code i.e. it's zoning at the ...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>'propertylandusetypeid'</td>
      <td>Type of land use the property is zoned for</td>
    </tr>
    <tr>
      <th>37</th>
      <td>'propertyzoningdesc'</td>
      <td>Description of the allowed land uses (zoning)...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>'rawcensustractandblock'</td>
      <td>Census tract and block ID combined - also con...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>'censustractandblock'</td>
      <td>Census tract and block ID combined - also con...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>'regionidcounty'</td>
      <td>County in which the property is located</td>
    </tr>
    <tr>
      <th>41</th>
      <td>'regionidcity'</td>
      <td>City in which the property is located (if any)</td>
    </tr>
    <tr>
      <th>42</th>
      <td>'regionidzip'</td>
      <td>Zip code in which the property is located</td>
    </tr>
    <tr>
      <th>43</th>
      <td>'regionidneighborhood'</td>
      <td>Neighborhood in which the property is located</td>
    </tr>
    <tr>
      <th>44</th>
      <td>'roomcnt'</td>
      <td>Total number of rooms in the principal residence</td>
    </tr>
    <tr>
      <th>45</th>
      <td>'storytypeid'</td>
      <td>Type of floors in a multi-story house (i.e. b...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>'typeconstructiontypeid'</td>
      <td>What type of construction material was used t...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>'unitcnt'</td>
      <td>Number of units the structure is built into (...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>'yardbuildingsqft17'</td>
      <td>Patio in  yard</td>
    </tr>
    <tr>
      <th>49</th>
      <td>'yardbuildingsqft26'</td>
      <td>Storage shed/building in yard</td>
    </tr>
    <tr>
      <th>50</th>
      <td>'yearbuilt'</td>
      <td>The Year the principal residence was built</td>
    </tr>
    <tr>
      <th>51</th>
      <td>'taxvaluedollarcnt'</td>
      <td>The total tax assessed value of the parcel</td>
    </tr>
    <tr>
      <th>52</th>
      <td>'structuretaxvaluedollarcnt'</td>
      <td>The assessed value of the built structure on t...</td>
    </tr>
    <tr>
      <th>53</th>
      <td>'landtaxvaluedollarcnt'</td>
      <td>The assessed value of the land area of the parcel</td>
    </tr>
    <tr>
      <th>54</th>
      <td>'taxamount'</td>
      <td>The total property tax assessed for that asses...</td>
    </tr>
    <tr>
      <th>55</th>
      <td>'assessmentyear'</td>
      <td>The year of the property tax assessment</td>
    </tr>
    <tr>
      <th>56</th>
      <td>'taxdelinquencyflag'</td>
      <td>Property taxes for this parcel are past due as...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>'taxdelinquencyyear'</td>
      <td>Year for which the unpaid propert taxes were due</td>
    </tr>
  </tbody>
</table>
</div>



## Visualization summary 

### 1. Does the number of bedrooms and bathrooms in a house increase the cost of the home
- It appears that the number of bedrooms have a positive relationship with the cost of the price of house
- Additionally according to the charts the higher the number of bathrooms the higher the price

### 2. Is the price of the house influenced by the county/ location?
- It appears that houses are less expensive in LA and than in Ventura and Orange county respectively
- However this does not seem like a strong relationship with the price, therefore this feature will not be  explored in this report. I will revisit after

### 3. Does the age of the house matter when it comes to the price? 
- The houses in Ventura and in Orange county appear to be more expensive the newer they are, unlike in Los angeles

### 4. Does the size of the house matter when it comes to the price? 
- It appears that the size of the house has a high positive relationship with the price. This is also evident with the positive relationship between the number of bedrooms and total_taxes

#### In summary I will explore statistical relationships in bathrooms, bedrooms, the year built and square feet(size)

### Hypothesis Testing
### 1st Hypothesis
#### $ H1 $ : the higher the number of bathrooms, the higher the price of the house 
### 2nd Hypothesis
#### $ H1 $ : the higher the number of bedrooms, the higher the price of the house 
### 3rd Hypothesis
#### $H1$:the newer the property the more expensive it is 
### 4th Hypothesis
#### $H1$: the size of the house in squarefeet has a relatively strong correlation with the property prices

### Summary hypothesis testing
- All of the variables had a positive correlation
- However I will proceed to model with three of the strongest correlated variables 
- These are:
    - bedrooms
    - bathrooms 
    - squarefeet

# Scaling
- #### Used minmax scaler from the preparefile
- #### function: prepare.scaling_zillow(train, validate, test, columns_to_scale)
 

# Modeling

## Baseline
- #### from the scaled data I split the data frame into X and y variablesto use them in modeling the baseline
- #### next, I convert them to dataframes in order to use dataframe functionality for the prediction proccess
- #### in step 1 and 2, I compute for the mean and median prediction
- #### lastly, I obtain the RMSE for the mean and the median for the baseline prediction

### MODELING (TweedieRegressor)
- #### the first step is create the model object withthe TweedieRegressor
- #### select the columns you would like to use in the modeling
- #### fit the model nto the training data 
- #### predict the train 
- #### evaluate the RMSE and compare it to the baseline

## MODELING (features: bedrooms, bathrooms, squarefeet)
#### Best model is a linear regression with 3 features
- #### the first step is create the model object with linear regression
- #### second, I select the columns I would like to use in the modeling
- #### fit the model into the training data 
- #### predict the train 
- #### evaluate the RMSE and compare it to the baseline

### Summary
- the linear regression shows the most promise with these three features.
- however there are concerns that these features are very similar or the bedrooms and the bathrooms are contained int the sqfeet
- so far this is the best model

### Baseline

RMSE using Mean

Train/In-Sample:  264487.8

Validate/Out-of-Sample:  264650.06

RMSE using Median

Train/In-Sample:  268181.81

Validate/Out-of-Sample:  268086.55

### Linear regression

RMSE for OLS using LinearRegression

Training/In-Sample:  226874.71769377927 

Validation/Out-of-Sample:  224745.86491873223


### Conclusion
In conclusion, the model was able to beat the baseline by atleast $40000.

However, some of the features involved seem to have similar effect on the model.
The correlation coefficient of bathrooms and squarefeet was at 81. 

In the future, we can include some feature engineering where we would combine sqfeet, bedrooms and bathrooms to make one feature.

Additionally, with more time we can run other regression models to make the process more accurate 


```python

```
