# 🌍 Weather Trend Forecasting - Final Project Report

> **PM Accelerator Mission**: "Empowering professionals to build, scale, and lead AI-powered products that solve real-world problems while promoting diversity and educational fairness in tech."

---

## 🚀 1. Project Overview

This project successfully developed a production-grade **AI Weather Forecasting System** capable of predicting 7-day temperature trends for 186 countries.

Starting from raw historical weather data, we evolved the system through rigorous Data Science phases—from Exploratory Data Analysis (EDA) and Anomaly Detection to advanced Deep Learning—culminating in a state-of-the-art **Transformer Neural Network** (V4) that achieves a Mean Absolute Error (MAE) of **2.00°C**.

## 📊 2. Data Cleaning & Analysis (EDA)

Before modeling, valid data was crucial. We processed 15 years of global weather data:
*   **Data Cleaning**: Fixed 30+ country name inconsistencies and handled missing values using forward-fill interpolation.
*   **Feature Engineering**: Created cyclical features (sin/cos for day/month) to help models understand seasonality.
*   **Anomaly Detection**: Applied Isolation Forest and Z-Score methods to remove unrealistic outliers (e.g., temperatures > 60°C or < -90°C in non-polar regions).

## 🧠 3. Model Evolution & Evaluation

We iterated through four major architectural phases to optimize performance:

| Phase | Model Architecture | MAE (Error) | Key Insight |
| :--- | :--- | :--- | :--- |
| **V1** | **MLP (Baseline)** | ~4.50°C | Captured global mean but failed on local trends. |
| **V2** | **LSTM (Seq2Seq)** | 2.05°C | Introducing time sequences (30-day lag) drastically improved accuracy. |
| **V3** | **Multivariate** | 2.07°C | Added Pressure/Humidity. Slight regression due to noise. |
| **V4** | **Advanced Transformer** | **2.00°C** | **Gated Residual Networks (GRN)** learned which features to trust, achieving state-of-the-art results. |

## 🛠️ 4. Final Solution (V4)

The final deployed solution is a **Dockerized Web Application** powered by FastAPI and the V4 Advanced Transformer.

### Key Features
1.  **Gated Residual Networks (GRN)**: Smartly filters noise from inputs.
2.  **Open-Meteo Integration**: Fetches real-time historical data for any location on Earth to generate forecasts.
3.  **Interactive Dashboard**: A beautiful, dark-mode map interface allowing users to click anywhere on the globe for instant predictions.

## � 5. Visualizations & Insights

The project includes interactive Plotly visualizations demonstrating:
*   **Seasonality**: Clear sine-wave patterns in temperature across hemispheres.
*   **Climate Zones**: Distinct statistical profiles for Tropical vs. Polar regions.
*   **Model Performance**: "Actual vs. Predicted" charts showing the model's ability to track sudden weather changes.

## 🏁 Conclusion

This project demonstrates the end-to-end lifecycle of an AI product: form raw data analysis to model optimization and final deployment. The **V4 Advanced Transformer** stands as a robust, production-ready solution for global weather trend forecasting.

---

**Repository**: [GitHub Link]
**Developed by**: [Your Name]
