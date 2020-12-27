import numpy as np
import pandas as pd

import re

from PIL import Image

import pickle

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib import pyplot

from lightgbm import LGBMRegressor

import streamlit as st


DATASET_URL = "https://www.kaggle.com/volpatto/coffee-quality-database-from-cqi"
DEST_ON_PC = "C:/Users/Artem/Desktop/ML/HW4"
DEST = "."

def classify(pred: float) -> str:
	if pred < 80:
		return "Underscorer"
	elif pred < 85:
		return "Very good"
	elif pred < 90:
		return "Excellent"
	return "Outstanding"


def year_transformer(x):
    if isinstance(x, str):
        x = re.sub(r"[^0-9]+", "", x)
        x = re.sub(r"[^0-9] ", "", x)
        if x:
            return int(x[:4])


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
	df_clean = df.copy()
	df_clean.drop(df_clean[df_clean["Total.Cup.Points"] < 70].index, inplace=True)
	df_clean["Harvest.Year"] = df_clean["Harvest.Year"].apply(
	    lambda x: year_transformer(x))
	df_clean["Expiration"] = df_clean["Expiration"].apply(
	    lambda x: year_transformer(x))
	df_clean["Grading.Date"] = df_clean["Grading.Date"].apply(
	    lambda x: year_transformer(x))
	num_cols = df_clean.select_dtypes(exclude='object').columns
	cat_cols = df_clean.select_dtypes(include='object').columns
	df_clean[cat_cols] = df_clean[cat_cols].apply(
	    lambda col: col.fillna(col.mode()[0]))
	df_clean[num_cols] = df_clean[num_cols].apply(
	    lambda col: col.fillna(col.median()))
	for col in cat_cols:
	    df_clean[col] = df_clean[col].apply(lambda x: re.sub("[^a-zA-Z]+", "", x.lower()))
	return df_clean, num_cols, cat_cols


def transform_data(df, num_cols, cat_cols):
	transformed_df = df.copy()
	for col in cat_cols:
		transformed_df[col] = transformed_df[col].astype('category')
		transformed_df = pd.concat([transformed_df.drop(col, axis=1),
	    						    pd.get_dummies(transformed_df[col],
	    						    prefix=col)], axis=1)
		transformed_df[num_cols] = transformed_df[num_cols].apply(lambda x: np.log(x + 1))
	num_cols_copy = []
	for col in num_cols:
		if col != "Total.Cup.Points":
			num_cols_copy.append(col)
	scaler = MinMaxScaler()
	transformed_df[num_cols_copy] = scaler.fit_transform(transformed_df[num_cols_copy])
	return transformed_df.drop(["Total.Cup.Points"], axis=1)


@st.cache()
def load_data(filepath: str) -> pd.DataFrame:
	df = pd.read_csv(filepath)
	return df


def slidebar(name: str, min_v: float, max_v: float, step: float) -> st.sidebar.slider:
	aver_v = (max_v + min_v) / 2
	return st.sidebar.slider(f"Choose your {name}: ", min_value=min_v, max_value=max_v, value=aver_v, step=step)


def main():
	df = load_data(f"{DEST}/merged_data_cleaned.csv")
	df_copy = df[['Country.of.Origin', 'Harvest.Year', 'Region', 'Company', 'Owner',
               "Total.Cup.Points", 'Variety', 'Processing.Method', 'Aroma', 'Expiration',
               'Grading.Date', 'Moisture', 'Quakers', 'unit_of_measurement',
               'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity',
               'Sweetness', 'Category.One.Defects', 'Color', 'Category.Two.Defects',
               'altitude_mean_meters', 'Certification.Body']]
	model = pickle.load(open(f"{DEST}/repr_model.pickle", "rb"))
	choice = st.selectbox("Content choice", ["Descriptive part", "Data review", "Representative model", "Conclusion"])
	if choice == "Descriptive part":
		image = Image.open(f'{DEST}/flavors.jpeg')
		st.markdown("## Foreword")
		col1, col2 = st.beta_columns(2)
		col1.markdown("### The gastronomic cases are clearly defined by interest of placeholders.\
    		Being apart from the buisness but passionate about coffee and inspired by the development\
    		of a branch of so-called 'third-wave' coffee in Berlin I am starting this little mostly\
    		exploratory research. I suppose the results could be beneificial for a person planning own\
    		business in the sphere of 'Kaffee-Rösterei'	as a part of a more global work to predict the\
    		 behaviour of customers. Analysis iof customers' recipts would help to form the supply work of a cafe.\
    		So with this description we proceed to the mechanism of predicting the score a sort of coffee\
    		would get evaluating some features.")
		col2.image(image, caption="The flavors variety", width=350,)
		st.sidebar.markdown("## About the phenomenon")
		st.sidebar.markdown("\n")
		st.sidebar.markdown("\n*** The third wave coffee movement is a movement led by both consumers and manufacturers \
			to consume, enjoy, and appreciate high-quality coffee (Wikipedia)***\n")
		st.sidebar.markdown("*** The first wave peaked when freeze-dried techniques made coffee popular, if not necessarily \
			any good. The second wave came with global Starbucksification, whereby large chains of gourmet coffee shops, home \
			espresso machines and the shift from robusta to aribica coffee beans - all helped improve coffee quality. 'The third \
			wave is about taking coffee to the next level' (The Guardian)***\n")
		st.sidebar.markdown("*** Third wave coffee treats coffee beans as an artisanal ingredient and seeks to convey the flavor in \
			the brewed coffee. (also Wikipedia)***\n")
	elif choice == "Data review":
		st.sidebar.markdown("### About the datasheet")
		st.sidebar.markdown("\n")
		st.sidebar.markdown("*** The database has an plethora of capabilities. Users can:***")
		st.sidebar.markdown("#### - Submit a coffee for Q Grading")
		st.sidebar.markdown("#### - View certified Q Coffees")
		st.sidebar.markdown("#### - Contact an owner of a Q Coffee")
		st.sidebar.markdown("#### - Register for a course")
		st.sidebar.markdown("#### - Register for a re-take")
		st.sidebar.markdown("#### - View and print Q certificates")
		st.sidebar.markdown("#### - Access high-resolution logos")
		st.sidebar.markdown("#### - View Q Graders and Q Instructors")
		st.sidebar.markdown("#### - Contact users within the database")
		st.sidebar.markdown("#### - Fill out a Coffee Corps™ application")
		st.sidebar.markdown("\n")
		st.sidebar.markdown("*** Coffee Quality Institute (CQI) is a non-profit organization working \
			internationally to improve the quality of coffee and the lives of people who produce \
			it. (The CQI about itself)***\n")
		st.markdown("## Review")
		st.markdown("### Here we get a glance on the information available for analysis")
		st.write(df_copy.sample(20))
	elif choice == "Representative model":
		image = Image.open(f'{DEST}/coffeebelt.jpg')
		st.markdown("### Here are some parameters that could be changed to get a better understanding how influancial each of them is \
					for scoring. On the left sidebar you could find parameters that are defined only by the official certification body")
		st.sidebar.markdown("### Features available after cupping")
		flavor = slidebar("Flavor", .0, 10., .01)
		aftertaste = slidebar("Aftertaste", .0, 10., .01)
		acidity = slidebar("Acidity", .0, 10., .01)
		body = slidebar("Body", .0, 10., .01)
		balance = slidebar("Balance", .0, 10., .01)
		uniformity = slidebar("Uniformity", .0, 10., .01)
		sweetness = slidebar("Sweetness", .0, 10., .01)
		col1, col2, col3 = st.beta_columns(3)
		with col1:
			origin = st.selectbox("Country of origin", df_copy[df_copy["altitude_mean_meters"] > 499]["Country.of.Origin"].unique())
		with col2:
			altitude = st.selectbox("Altitude", df_copy[(df_copy["Country.of.Origin"] == origin) &
														(df_copy["altitude_mean_meters"] > 499)]["altitude_mean_meters"].unique())
		with col3:
			moisture = st.slider("Moisture", .0, .3, .001)
		user_data = {"Flavor" : flavor,
					 "Aftertaste" : aftertaste,
					 "Acidity" : acidity,
					 "Body" : body,
					 "Balance" : balance,
					 "Uniformity" : uniformity,
					 "Sweetness" : sweetness,
					 "Country.of.Origin": origin,
					 "altitude_mean_meters": altitude,
					 "Moisture": moisture,}
		representative_row = pd.Series(data=user_data, name='Your data')
		col1, col2 = st.beta_columns([7, 6])	
		col1.write(representative_row)
		col2.image(image, use_column_width=True)
		user_dict = {}
		for col in df_copy.columns.to_list():
			user_dict[col] = np.nan
		user_dict.update(user_data)
		new_row = pd.DataFrame(data=user_dict, index=df.columns)
		df_cleaned = df_copy.append(new_row, ignore_index=False)
		df_cleaned, num_cols, cat_cols = preprocess_data(df_cleaned)
		transformed_user_data = transform_data(df_cleaned, num_cols, cat_cols)
		prediction = model.predict(np.array(transformed_user_data.iloc[len(transformed_user_data) - 1]).reshape(1, -1))
		predicted_class = classify(prediction)
		st.markdown(f"### For the given parameters the score estimation is")
		col1, col2, col3 = st.beta_columns(3)
		col2.markdown(f"## {round(prediction[0], 5)}")
		st.markdown(f"### which means that this sort would be classified as")
		col1, col2, col3 = st.beta_columns(3)
		col2.markdown(f"## {predicted_class}")
	elif choice == "Conclusion":
		st.sidebar.markdown("## The credits / contributions")
		st.sidebar.markdown("*** The materials used in process:***")
		st.sidebar.markdown("*** Wikipedia***")
		st.sidebar.markdown("*** CQI database for the year 2018***")
		st.sidebar.markdown("*** Some random fotos from Google***")
		st.sidebar.markdown("*** The Guardian article from the year 2009***")
		st.sidebar.markdown("*** LightGBM Regressor model***")
		col1, col2 = st.beta_columns([7, 5])
		col1.markdown("## The results")
		col1.markdown("### It is clear that some features are much more influencial on the estimation process as the others.\
					As we can see the most important is the conclusion of certification body. The results presented by \
					comission defines the further destiny of the sort. However some features being not so important \
					for forming the general score have some (not very strong) impact on the result. Representing \
					the interests of a business owner it should be also considered as a factor of risk minimization. \
					Beeing able to discuss the problem with some Rösterei-owners I have heard the opinion that most farmers and \
					roasters yearly show the quality not worse than in the previous season. Taking this idea as a starting point \
					we can see how important the minor attributes could be for predicting some estimation nuances.")
		col2.markdown("## Further development")
		col2.markdown("### There is much to do to make this applcation more useful. As I see it now potential directions \
					are the combining the data of previous seasons' estimations with some climatic features of the country \
					of origin or research of the germ population and adding some anomaly detection. Of course \
					all of the said above is only my personal opinion but making a little bit deeper analysis and communicating \
					a lot with business-owners it would be possible to contribute to 'third wave of coffee' development." )
		st.markdown("# THANK YOU FOR YOUR ATTENTION!!!")


if __name__ == "__main__":
    main()
