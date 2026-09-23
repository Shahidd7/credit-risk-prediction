🏦 Credit Risk Default Prediction with Explainable AI  
📖 Overview  
When a bank gives out a loan, they take a risk. If the borrower doesn't pay it back (a "default"), the bank loses money. But if the bank is too strict and denies everyone, they make no money.  
This project solves that problem by using Machine Learning to predict the exact probability that a borrower will default before the bank gives them the money. Furthermore, if a loan is denied, federal law requires the bank to explain why. So, this project doesn't just predict risk—it also generates a visual explanation for every single decision using SHAP.

---

📊 The Dataset  
I used the Lending Club Loan Data dataset, which contains real-world loan applications, borrower credit histories, and final loan outcomes (Fully Paid vs. Charged Off). The raw dataset contains 151 columns of mixed data types (numbers, dates, and text categories).

---

🛠️ Step-by-Step Breakdown  
Here is exactly how I built this pipeline from scratch, explained simply:  
Step 1: Data Cleaning & Removing Empty Columns  
Machine Learning models cannot understand empty data. The raw dataset had 151 columns, many of which were completely empty or contained mostly missing values. I wrote code to automatically find and drop any column that was 100% null, cleaning up the dataset and making it easier to read.  
Step 2: Stopping the "Time Traveler" Problem (Data Leakage)  
In real life, you don't know if someone will default until after you give them the loan. However, the raw dataset included features that only happen after a loan is issued—like "total late fees paid" or "last payment amount."  
If I trained the model on these features, the model would "cheat" by looking into the future. This is called Data Leakage, and it is the #1 reason ML models fail in production. I carefully identified and dropped 20+ of these post-origination columns to ensure the model only uses information available at the moment of application.  
Step 3: Feature Engineering (Text to Numbers)  
ML models only understand numbers, not text or calendar dates.  
• I converted dates like earliestcrline into a number: "Years of Credit History."  
• I cleaned strings like " 36 months" into the number 36.  
• For text categories like Grade (A, B, C) and Purpose (debt consolidation, credit card), I used LightGBM's Native Categorical Support. Instead of creating dozens of new columns (One-Hot Encoding), LightGBM can natively learn from text categories if we format them correctly, saving memory and improving accuracy.  
Step 4: Chronological Train/Test Split  
Normally, data scientists split their data randomly. In finance, this is a crime! If you train on loans from 2018 to predict loans from 2015, you are using future information to predict the past.  
Instead, I sorted the data by issued (the date the loan was issued). I trained the model on older loans (the past) and tested it on newer loans (the future) to simulate real-world banking conditions.  
Step 5: Training LightGBM & Handling Imbalance  
In the real world, most people pay back their loans. Only about 19% actually default. If I just threw the data into a model, it would just guess "Good Loan" every time and be 81% accurate without actually learning anything.  
I used LightGBM (a powerful tree-based algorithm) and implemented a technique called scaleposweight. This forced the model to pay extra attention to the rare default cases. I also used Early Stopping, which tells the model to stop training the moment it starts memorizing the data (overfitting).  
Step 6: Fixing the Probabilities (Calibration)  
Because I forced the model to care about defaults, the model got scared and started predicting 40%+ default rates for everyone! The model's ranking was correct (it knew who was risky), but its probabilities were wrong.  
If a bank prices a loan based on a fake 40% risk, they will overcharge the customer and lose the deal. I fixed this using Isotonic Regression Calibration. I trained a secondary mathematical model that "squished" the inflated 43% predictions back down to match the real-world 19% default rate. Now, if the model says "10% chance of default," you can literally trust that number.  
Step 7: Explainable AI (SHAP)  
Banks cannot use "black box" models. If a loan is denied, federal law requires the bank to send the customer a letter explaining exactly why.  
I used a library called SHAP (SHapley Additive exPlanations). SHAP uses game theory to break down a single prediction and tell you exactly how much each feature contributed. For example: "The model predicted a 30% default risk because their Debt-to-Income ratio pushed the risk up by 10%, and their recent credit inquiries pushed it up by 8%." I generated global feature importance bars and local "Waterfall" charts for individual applicants.  
Step 8: Deployment with Streamlit  
Code in a Jupyter Notebook is great, but it isn't a product. I built an interactive web application using Streamlit.  
• Users can slide bars to input an applicant's Income, Loan Amount, Interest Rate, and Credit Grade.  
• The app instantly calculates the Probability of Default using the calibrated model.  
• The app renders a live SHAP Waterfall chart explaining exactly why the model made that decision.

---

📈 Results & Metrics  
Because the data is imbalanced (only 19% defaults), plain "Accuracy" is a terrible metric. I evaluated the model using credit industry standards:  
• KS-Statistic (0.36): The Kolmogorov-Smirnov test measures the maximum separation between Good and Bad loans. >0.30 is considered a strong model in finance.  
• PR-AUC (0.45): Precision-Recall Area Under Curve measures how well the model identifies the minority (default) class.

---

💻 Tech Stack  
• Language: Python  
• Data Manipulation: Pandas, NumPy  
• Machine Learning: LightGBM, Scikit-Learn  
• Explainability: SHAP  
• Visualization: Matplotlib  
• Deployment: Streamlit
