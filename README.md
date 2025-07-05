    Project topic: Calories Burnt Prediction using Machine Learning with Python
    
    Created by: Karanvir Singh (CO22338) of CSE 3rd Year
    

(1) Description of the project:

    I have developed a web application for calories burnt prediction using machine learning with python. It takes some parameters 
    such as your age , gender , exercise duration etc and then it evaluates the amount of calories you would burn. In addition, you 
    will be able to observe similar results and general information (according to the parameter values that you would enter into 
    application). The application focuses on predicting the number of calories burnt during physical exercise. The project involves 
    analyzing a dataset (calories.csv file) containing information about exercise patterns and corresponding calorie measurements. we 
    explore various regression models to predict calorie burn based on exercise features. We train and evaluate models such as Linear 
    Regression, Ridge Regression, Lasso Regression, Decision Tree Regression and Random Forest Regression. The evaluation metrics used to 
    assess the models performance include Mean Squared Error (MSE) and R-Squared Score. The model with the best performance can be 
    selected for further analysis. All models are saved using the Pickle library for future use. Any one of the trained models can be 
    loaded and used to predict calories burnt for new input data. This step highlights the importance of model deployment in real-world 
    applications. We can select any model for use in web application.


(2) Setting up the project and running the web application on your computer:

(a) After downloading and saving the project folder into a certain directory, open it on any IDE like VS Code.

(b) Open integrated terminal for accessing directory where folder is present.

(c) Install all packages at once mentioned in requirements.txt file present in the folder using "pip install -r requirements.txt".
    You can also install every package individually using command "pip install package_name".

(d) To run web application, use command "streamlit run app.py". After couple of seconds the WebApp pops up in your browser.
    You can also click on the first link (local URL) out of the two links that will be generated to open the web application.
    Web App Local URL: http://localhost:8501 (This link will work after you have executed the above mentioned command which makes 
    sure that streamlit is running.)


(3) Description of the files included in the folder:

(a) app.py: File that contains source code for web application.

(b) Model_Training_Code.ipynb: File that contains source code for machine learning models used in project.

(c) calories.csv: File containing dataset for exercise patterns and corresponding calorie measurements.

(d) README.md: File that contains necessary information regarding project.

(e) requirements.txt: File that contains all necessary packages with their versions that need to be installed for successfull 
    implementation of project.

(f) rfr.pkl: This file pickles random forest regression model which is selected for use in web application.

(g) lr.pkl: This file pickles linear regression model which is selected for use in web application.

(h) ls.pkl: This file pickles lasso regression model which is selected for use in web application.

(i) dtr.pkl: This file pickles decision tree regression model which is selected for use in web application.

(j) rd.pkl: This file pickles ridge regression model which is selected for use in web application.

(k) X_train.csv: File containing dataset on which the selected model is trained.


Web App Link: https://co22338-calories-burnt-predictor.streamlit.app/
