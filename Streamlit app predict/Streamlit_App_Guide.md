
# Walkthrough - Streamlit App Deployment

I have created a Streamlit application to demonstrate the machine learning model in a user-friendly web interface.

## Files Created
1. `save_model.py`: A script to extract the preprocessing pipeline and trained Random Forest model from the notebook and save them as `.pkl` files.
2. `app.py`: The main Streamlit web application.
3. `requirements.txt`: A list of python dependencies.
4. `*.pkl`: Serialized model artifacts (`model.pkl`, `preprocessor.pkl`, `pca.pkl`, `columns_info.pkl`).

## How to Run the App
1.  **Install Requirements** (if not done):
    ```bash
    pip install -r requirements.txt
    ```
2.  **Generate Model Artifacts** (if not done):
    ```bash
    python save_model.py
    ```
3.  **Launch the App**:
    ```bash
    streamlit run app.py
    ```

## Verification
I have automated a test of the application to ensure it loads and predicts correctly.

![Streamlit App Demo](streamlit_app_test_1765019734585.webp)

**Test Results**:
- App launched successfully on port 8501.
- Sidebar inputs for Customer Profile and Economic Indicators work.
- "Predict Subscription" button successfully calls the model pipeline.
- Prediction results (YES/NO) and confidence scores are displayed.
