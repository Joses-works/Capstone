import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# Load dataset
hs_df = pd.read_csv ("SomervilleHappinessSurvey.csv",encoding='UTF-16 LE')
st.set_page_config(
    page_title="Sommerville Happiness Survey",page_icon=" 🗿 🗿 🗿"
)
st.title("Happiness Survey")
st.write("This is my survey data.")
# Display dataset
st.write(hs_df)


# Q1
st.write('Question 1.(What is the mode rating for each of the six attributes (X1 to X6))')
st.write('This is a bar chart showing the mode ratings of all six attributes.')
st.write('As shown in the Graph you can see that X1 and X6 are the Attributes that are most positively percieved.')
mode_ratings = hs_df.mode().iloc[:, 1:]
data = {
        'X1': [3, 3, 5, 5, 5],
        'X2': [3, 2, 3, 4, 4],
        'X3': [3, 3, 3, 3, 3],
        'X4': [4, 5, 3, 3, 3],
        'X5': [2, 4, 3, 3, 3],
        'X6': [4, 3, 5, 5, 5]}
df = pd.DataFrame(data)


mode_ratings = df.mode().iloc[0]

fig = plt.figure()
mode_ratings.plot(kind='bar', figsize=(8, 6))
plt.xlabel('Attributes')
plt.ylabel('Rating')
st.pyplot(fig)

#QUestion 2
st.write('Question 2.(Which attribute (X1 to X6) has the strongest correlation with overall happiness?)')
st.write('This is a bar chart showing the correlation coefficients between each attribute and overall happiness..')
st.write('As shown in this graph you can see that X1(the availability of information about city services) is the attribute that correlates mostly with overall happiness.')
corr_matrix = hs_df.corr()


c_with_d = corr_matrix.iloc[0, 1:]

fig = plt.figure()
c_with_d.plot(kind='bar', figsize=(8, 6))
plt.title('Correlation with Overall Happiness')
plt.xlabel('Attribute')
plt.ylabel('Correlation Coefficient')
st.pyplot(fig)
#Q3
st.write('Question 3.(How does the availability of social events(X6) vary across different levels of happiness (D=0 or D=1)?)')
st.write('This is a bar chart showing the mean availability of social events for each happiness level.')
st.write('as shown in this graph,you can see that when more social events are available it leads to overall happiness.')
social_by_happiness = hs_df.groupby('D')['X6'].mean()

fig = plt.figure()
social_by_happiness.plot(kind='bar', figsize=(8, 6))
plt.title('Availability of Social Events')
plt.xlabel('Happiness Level')
plt.ylabel('Mean Availability of Social Events')
plt.xticks(rotation=0)
st.pyplot(fig)
#Q4
st.write('Question 4.(How does the cost of housing (X2) vary across different levels of happiness (D=0 or D=1)?)')
st.write('This is a box plot showing the distribution of housing costs for each happiness level.')
st.write('in this boxplot you can see that 2 and 3 are the most frequent rating for X2,you can also see that 5 is a very rare rating.The median value for people that are not happy is 3 and the median value for people who are happy is 2 meaning the higher the cost of housing the less happy people get.')
fig, ax = plt.subplots()
hs_df.boxplot(column='X2', by='D', grid=False,ax=ax)
plt.xlabel('Happiness level')
plt.ylabel('Cost of housing')
plt.title('Cost of housing by happiness level')
st.pyplot(fig)
#Q5
st.write('Question 5.(how does the maintenance of streets and sidewalks affect overall happiness (D=1)?)')
st.write('This is a bar chart showing the mean maintenance of sidewalks for each happiness level.')
st.write('As you can see in the graph on average people are happier when the sidewalks and roads are highly maintained.')
main = hs_df.groupby('D')['X5'].mean()
fig, ax = plt.subplots()
main.plot(kind='bar', figsize=(8, 6))
plt.title('maintence of streets and sidewalks affect on happiness')
plt.xlabel('0=unhappy,1=happy')
plt.ylabel('Mean maintenance')
plt.xticks(rotation=0)
st.pyplot(fig)
#Accuracy 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = hs_df.drop('D', axis=1)
y = hs_df['D']

#80% is used for training and 20% is used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

#to make predictions
y_pred = dt.predict(X_test)

#compares predicted vs actual values
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
#Accuracy chart
st.write('This charts compares its predicted values to the real values it gets from each percentage of data used for the experiment.')
import matplotlib.pyplot as plt

# Calculate accuracy for different test sizes
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
accuracies = []
fig, ax = plt.subplots()
for size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plot the accuracies
plt.plot(test_sizes, accuracies, marker='o')
plt.xlabel('Test Size(percentage)')
plt.ylabel('Accuracy')
plt.title('Accuracy chart based on percentage of data')
plt.grid(True)
st.pyplot(fig)
st.write('This here is the Visualization of the data')
fig = plt.figure(figsize =(18,18))
fig, ax = plt.subplots()
ax=fig.gca()
hs_df.hist(ax=ax,bins =30)
st.pyplot(fig)