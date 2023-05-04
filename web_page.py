import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# Load dataset
hs_df = pd.read_csv ("SomervilleHappinessSurvey.csv",encoding='UTF-16 LE')
st.title("Happiness Survey")
st.write("This is my survey data.")
# Display dataset
st.write(hs_df)

# Plot bar chart
st.write('Question 1.(What is the mode rating for each of the six attributes (X1 to X6))')
st.write('This is a bar chart showing the mode ratings of all six attributes.')
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
hs_df.boxplot(column='X2', by='D', grid=False)

fig = plt.figure()
plt.xlabel('Happiness level')
plt.ylabel('Cost of housing')
plt.title('Cost of housing by happiness level')
st.pyplot(fig)
#Q5
st.write('Question 5.(how does the maintenance of streets and sidewalks affect overall happiness (D=1)?)')
st.write('This is a bar chart showing the mean maintenance of sidewalks for each happiness level.')
main = hs_df.groupby('D')['X5'].mean()
fig, ax = plt.subplots()
main.plot(kind='bar', figsize=(8, 6))
plt.title('maintence of streets and sidewalks affect on happiness')
plt.xlabel('0=unhappy,1=happy')
plt.ylabel('Mean maintenance')
plt.xticks(rotation=0)
st.pyplot(fig)
