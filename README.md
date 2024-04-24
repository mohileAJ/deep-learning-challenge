# deep-learning-challenge

## Report on the Neural Network Model

### Overview of the analysis:  
The non-profit foundation, Alphabet Soup, wants a tool to run classification modeling to identify / predict whether funding applicant is a potentially good target for funding by Alphabet Soup. A dataset containing features and the target variable (Whether "Successful" or not) had been provided. 

Data Preprocessing: Using Pandas, such dataset containing the selected features along with the target variable, is loaded into DataFrame for merging and cleaning steps.  

Further, the 'IS_SUCCESSFUL' column is identified as our target variable (y-variable), while the selected features such as removing "EIN" amd "NAME" as these are not relevant for modeling; and selecting others based on importance (e.g., grouping the outliers in Application_Type and Classification, by using value_counts and binning) are also further processed. 

The above normalised dataset of features is used as input to fit the model to predict the target variable.  The dataset is split into training and testing sets (before it is used as input), using scikit-learn's StandardScaler.  

 Compile, Train, and Evaluate:  Using the same data that had been normalised in Step 1 above, created a neural network to train and test the data (in order to fit the model with the dataset of selected features), using TensorFlow and Keras.  The model architecture used to find a relationship between the input and output (by fitting the model on the data) is as follows: 
 - one input layer (25,724, 76), two hidden layers of size 30 neurons each; 
 - using Sequential / Dense model; 
 - relu activation for hidden layers, and sigmoid activation for the output layer; and
 - 100 epochs. 

The model structure included 3,271 parameters in total, which were all trainable. 

Evaluation was based on accuracy (which, unfortunately, was in the 71.34%), and loss (60.4%) metrics.  Experimented by trying different layer sizes, number of layers (removing one hidden layer), and different epochs at each runtime. Other experiments that could be done are to make even more structural changes to the model (such are the learning parameters, batch size, learning rate, etc.). 

Summary: 
I could have done PCA, given that the number of features are a bit too many in this dataset, and PCA would be useful to reduce the noise.  Also, logistic regression would have been a simpler classification approach and easier to understand. 


