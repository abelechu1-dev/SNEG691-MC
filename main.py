import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import classification_report



from sklearn.ensemble import RandomForestClassifier


from sklearn.linear_model import LogisticRegression


df = pd.read_csv("All Four.csv")
print(df.head())# prints the head of the dataset

print("Total rows:", len(df))
print("Total columns:", len(df.columns))


print("Print the data type of the collumns")
print(df.dtypes)#prints the data type of each column
print()

print("Update the colmn data type and print the types")
#Below we convert the time from string to actual timestamp
df["Time Stamp (yyyy-MM-dd HH:mm:ss.SSS)"] = pd.to_datetime(
    df["Time Stamp (yyyy-MM-dd HH:mm:ss.SSS)"]
)

print(df.dtypes)
print()

#Below code checks if there is a missing vlaue in our data
print(df.isnull().sum())
print()

#Below code checks of there is a duplicate
print("Duplicate rows:", df.duplicated().sum())
print()

#Below code adds a column called hour and get the vlaue from the timestamp and keep it in that column
df["Hour"] = df["Time Stamp (yyyy-MM-dd HH:mm:ss.SSS)"].dt.hour

print(df[["Hour"]].head())
print()

#Below code creates a column called days of week it get the day and stores the vlaue in that column
df["DayOfWeek"] = df["Time Stamp (yyyy-MM-dd HH:mm:ss.SSS)"].dt.day_name()

print(df["DayOfWeek"].head())
print()

#Below code returns if the day we got earlier is a weekday or weekend and store it in the column called IsWeekend
df["IsWeekend"] = (df["Time Stamp (yyyy-MM-dd HH:mm:ss.SSS)"].dt.day_of_week >= 5).astype(int)
print(df["IsWeekend"].head())
print()

#Calculates the average, minimum, and maximum noise level for each building.
print(df.groupby("Location")["Recorded Value (dBA)"].agg(["mean", "min", "max"]))

print("\n")

def classify_noise(dba):
    if dba < 40:
        return "Quiet"
    elif dba <= 60:
        return "Moderate"
    else:
        return "Loud"

df["NoiseCategory"] = df["Recorded Value (dBA)"].apply(classify_noise)

print(df["NoiseCategory"].value_counts())


print(2*"\n")


X = df[["Hour", "DayOfWeek", "IsWeekend", "Location", "Time"]]
X = pd.get_dummies(X)

y = df["NoiseCategory"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training:", len(X_train))
print("Testing:", len(X_test))

print()

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("Decision Tree Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
print("Decision Tree F1 Score:", f1_score(y_test, y_pred, average="weighted", zero_division=0))
print(classification_report(y_test, y_pred, zero_division=0))

print()

print("Rnaodm Forest")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n===== RANDOM FOREST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred, average="weighted", zero_division=0))
print("F1 Score:", f1_score(y_test, rf_pred, average="weighted", zero_division=0))
print(classification_report(y_test, rf_pred, zero_division=0))

print()

print("Regression")
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n===== LOGISTIC REGRESSION RESULTS =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred, average="weighted", zero_division=0))
print("F1 Score:", f1_score(y_test, lr_pred, average="weighted", zero_division=0))
print(classification_report(y_test, lr_pred, zero_division=0))