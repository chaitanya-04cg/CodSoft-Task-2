import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#dataset
data = pd.read_csv("rating.csv")

#features to check
features = ['TV', 'Radio', 'Newspaper']

#column
y = data['Sales']


for feature in features:

    print("\nFeature:", feature)

   
    X = data[[feature]]

    #remove missing values
    final_data = pd.concat([X, y], axis=1)
    final_data.dropna(inplace=True)

    X = final_data[[feature]]
    y = final_data['Sales']

    #split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    #create model
    model = LinearRegression()

    #train model
    model.fit(X_train, y_train)

    # Predict values
    pred = model.predict(X_test)

    #print score
    print(
        "R2 Score:",
        r2_score(y_test, pred)
    )

    #graph
    plt.figure(figsize=(6,4))
    plt.scatter(X_test, y_test)
    plt.plot(X_test, pred)
    plt.xlabel(feature)
    plt.ylabel("Sales")
    plt.title(feature + " vs Sales")
    plt.show()