# settings for testing

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ONLY RELEVANT FOR BINARY CLASSIFICATION: threshold for the average label of the training instances for a single case in order for the case to be labelled '1' rather than '0'
# 0.5 is a reasonable value for general use
threshold = 0.5

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# choose which sets to use for making predictions and then analyzing
# it's good to use both train and val - this will allow you to make comparisons, understand existing overfitting, etc.

pred_train = True

pred_val = True

# the reserved test set should be left alone for the most part, and only be used to test promising models
pred_test = False


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------