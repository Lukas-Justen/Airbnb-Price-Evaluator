# Airbnb-Price-Evaluator 

![alt text](https://cdn-images-1.medium.com/max/1600/1*yZ1LPIcXnnW6Ubmp2M-0rQ.png "Airbnb")

Our group will try and address the task of understanding Airbnb market dynamics. There are several different ways to accomplish this goal, but more specifically, we will try and predict the best price for any Airbnb given standard measures such as the location of the listing, the date of pricing, and the features that particular Airbnb offers. This can take several different forms as our project unfolds. Our most promising idea is to try to recommend a price for an individual looking to put a listing on Airbnb for the first time. However, we know that reviews play a strong part of customer perception, so we would also like to explore if any given Airbnb is undervalued or overvalued given their historical performance.

Datset:
-------
For our project we used public available Airbnb datasets that can be found on Kaggle. The datasets offer data for the city of Seattle and Boston. We decided to use these sets since the data for both cities nearly had the same structure. This made it easy for us to integrate both datasets and make them usable for our project. Finally, we also found a similar dataset for New York. This dataset is much bigger compared to the other to cities and might create an unbalance in our data. However, we thought that it makes sense to use a subset of the New York dataset in order to increase diversity and create a more generalized model:

__Boston__ (3586 rows): https://www.kaggle.com/airbnb/boston 

__Seattle__ (3818 rows): https://www.kaggle.com/airbnb/seattle 

__New York__ (44317 rows): https://www.kaggle.com/peterzhou/airbnb-open-data-in-nyc 

Process:
--------
1. __Basic__  
   At this stage we want to use the information each listing offers without the need for reviews. This will allow us to gather some insight into the features our data provides. Moreover, this will help us to assess the importance of specific features without mixing it with data that might cloud the results. On the one hand, the datasets contains information about the host and his verifications. On the other hand, we have information about the number of beds, the listings geolocation or the ameneties it provides. These features will help us to get a first insight into the pricing landscape.
2. __Advanced__  
   Since we think that reviews for listings are important in order to assess if its price is reasonable we want to use the different types of reviews at this stage. The datasets offer different review categories like communication or location. In addition, the datasets gives us information about the number of reviews or when the first and last review has been made.
3. __Deep__  
   The last stage for this project includes text analysis or image analysis with deep neural networks. Here we want to focus on feature engineering in order to find new predictive features about the reasonability of a listing's price. Due to time constraints this stage might be too much for our timeframe. However, we plan to continue with our work on this project. Thus, this step is crucial in order to increase accuracy of our model. Moreover, this helps to get a better accuracy on the cold start of a listing that has no reviews.

The following table shows the features we want to use in each of the stages. Each stage enhances the list of features of the previous stages:

| Basic | Advanced | Deep|
|------|--------|-------|
|host_id|review_scores_location|description|
|host_since|review_scores_value|picture_url|
|host_response_time|review_scores_communication|host_picture_url|
|host_response_rate|review_scores_rating
|host_acceptance_rate|review_scores_accuracy
|host_is_superhost|review_scores_cleanliness
|host_total_listings_count|review_scores_checkin
|host_verifications|reviews_per_month
|host_identity_verified|number_of_reviews
|latitude|first_review
|longitude|last_review
|property_type|
|room_type|
|accommodates|
|bathrooms|
|bedrooms|
|beds|
|bed_type|
|amenities|
|guests_included|
|minimum_nights|
|cancellation_policy|
|__22 new features__|__11 new features__|__3 new features__|


Algorithms:
----------

Performance:
-----------
Peer Reviews:
------------
- Do we need a cloud instance for training because of the dataset size?
- What is our benchmark or to what is the groundtruth for evaluating the model?
- How do we plan to test the accuracy of our model?
- Does our dataset contain first listing prices and how to we use them?
- Do we plan to use KNNs for Clustering similar listings based on features?
- How doe we know if the predicted price is correct or wrong?
- Which datasets can we add to our project and where are they from?
- Do we have geographic information that might help calculating the price?


_by Keith Pallo, Rhett Dsouza, Albert Z. Guo, Lukas Justen_

