import pprint

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

# get Katie's net worth (she's 27)
# sklearn predictions are returned in an array, so you'll want to index into
# the output to get what you want, e.g. net_worth = predict([[27]])[0][0] (not
# exact syntax, the point is the [0] at the end). In addition, make sure the
# argument to your prediction function is in the expected format - if you get
# a warning about needing a 2d array for your data, a list of lists will be
# interpreted by sklearn as such (e.g. [[27]]).
# km_net_worth = 1.0 ### fill in the line of code to get the right value
km_net_worth = reg.predict([[27]])[0][0]

# get the slope
# again, you'll get a 2-D array, so stick the [0][0] at the end
# slope = 0. ### fill in the line of code to get the right value
slope = reg.coef_[0][0]

# get the intercept
# here you get a 1-D array, so stick [0] on the end to access
# the info we want
# intercept = 0. ### fill in the line of code to get the right value
intercept = reg.intercept_[0]

# get the score on test data
test_score = 0.  ### fill in the line of code to get the right value
test_score = reg.score(ages_test, net_worths_test)

# get the score on the training data
training_score = 0.  # fill in the line of code to get the right value
training_score = reg.score(ages_train, net_worths_train)


def submitFit():
    # all of the values in the returned dictionary are expected to be
    # numbers for the purpose of the grader.
    return {"networth": km_net_worth,
            "slope": slope,
            "intercept": intercept,
            "stats on test": test_score,
            "stats on training": training_score}


if __name__ == "__main__":
    pprint.pprint(submitFit())