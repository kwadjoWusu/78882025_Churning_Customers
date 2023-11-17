# 78882025_Churning_Customers
This program predicts whether or not the customer is churning or not
It uses a Neutral Network in training to predict whether churning
It uses the 10 best features (Contract','OnlineSecurity', 'PaymentMethod', 'TechSupport', 'gender', 'InternetService', 'OnlineBackup','MonthlyCharges','TotalCharges', 'tenure') from the feature extraction to train the model. The optimized hyperparamters of the model where 'batch_size': 40, 'epochs': 40, 'optimizer': 'adam', I use a dropout at every layer in the hidden to prevent over fitting. The dropout percentage of 0.4. The problem is a binary classification problem so i use a sigmoid function. My AUC score for my best model was 71.49% 
