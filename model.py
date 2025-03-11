import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    print("GPU detected. Configuring for maximum GPU performance.")
else:
    print("No GPU detected. Running on CPU.")
# Load dataset and clean column names
def load_data_xlsx(file_path, sheet_name="Data Set with Labels Text"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Rename columns for clarity
    df.rename(columns={
        "Q16A. What is the most important thing you LIKE about the shown concept}?     This can include anything you would want kept for sure or aspects that might drive you to buy or try it…       Please type a detailed response in the space below":
            "Q16A_Likes",
        "Q16B. What is the most important thing you DISLIKE about the shown concept}?    This can include general concerns, annoyances, or any aspects of the product that need fixed for this to be more appealing to you...     Please type a detailed response in the space below.":
            "Q16B_Dislikes",
        "OE_Quality_Flag": "Quality_Flag"
    }, inplace=True)
    
    # Get beer preference columns
    beer_preference_columns = [col for col in df.columns if col.startswith('Q18_')]
    print(f"Found {len(beer_preference_columns)} beer preference columns: {beer_preference_columns}")
    
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df, beer_preference_columns

# Prepare text features - enhanced to create separate features
def prepare_text_features(row):
    # Create separate features for likes and dislikes
    text_likes = str(row["Q16A_Likes"]) if pd.notna(row["Q16A_Likes"]) else "missing_text"
    text_dislikes = str(row["Q16B_Dislikes"]) if pd.notna(row["Q16B_Dislikes"]) else "missing_text"
    return text_likes, text_dislikes

# Main function
def main(file_path, sheet_name):
    df, beer_preference_columns = load_data_xlsx(file_path, sheet_name)
    
    # Drop rows with missing Quality_Flag and reset index
    df = df.dropna(subset=["Quality_Flag"]).reset_index(drop=True)
    
    # Convert Quality_Flag to binary labels
    df["Quality_Flag_Binary"] = df["Quality_Flag"].apply(lambda x: 1 if x in [1, '1.0'] else 0)
    
    # Prepare text features - now creating separate features for likes and dislikes
    print("Preparing text features...")
    df["Likes_Text"], df["Dislikes_Text"] = zip(*df.apply(prepare_text_features, axis=1))
    
    # Encode text features using SBERT
    print("Encoding text with SBERT...")
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    likes_embeddings = sbert_model.encode(df["Likes_Text"].tolist(), batch_size=16)
    dislikes_embeddings = sbert_model.encode(df["Dislikes_Text"].tolist(), batch_size=16)
    
    # Combine embeddings
    embeddings = np.hstack((likes_embeddings, dislikes_embeddings))
    
    # Count class distribution
    class_counts = df["Quality_Flag_Binary"].value_counts()
    print(f"\n* Class distribution in dataset:")
    print(f"   - Good responses (0): {class_counts.get(0, 0)}")
    print(f"   - Bad responses (1): {class_counts.get(1, 0)}")
    
    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, 
        df["Quality_Flag_Binary"].values, 
        test_size=0.2, 
        random_state=42, 
        stratify=df["Quality_Flag_Binary"]
    )
    
    print(f"\n* Training set size: {len(X_train)}")
    print(f"* Test set size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to balance classes
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, sampling_strategy=0.7)  # Increase minority class representation
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Convert to numpy arrays for proper counting
    print(f"Original class distribution - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
    print(f"Resampled class distribution - Class 0: {np.sum(y_resampled == 0)}, Class 1: {np.sum(y_resampled == 1)}")
    
    # Define base models with optimized hyperparameters
    print("Setting up base models...")
    base_models = [
        ('lr', LogisticRegression(class_weight='balanced', C=0.8, solver='liblinear', max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, min_samples_split=4, random_state=42)),
        ('svm', SVC(probability=True, class_weight='balanced', kernel='rbf', C=1.0, gamma='scale', random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance')),
        # Removed XGBoost due to memory error
        ('gb', GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=4, subsample=0.8, random_state=42))
    ]
    
    # Create stacking ensemble
    print("Creating stacking ensemble model...")
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(class_weight='balanced', C=1.2, solver='liblinear', random_state=42),
        cv=5,  # 5-fold cross-validation
        stack_method='predict_proba',
        n_jobs=-1  # Use all available cores
    )
    
    # Fit the stacking model
    print("Training stacking ensemble...")
    stacking_model.fit(X_resampled, y_resampled)
    
    # Make predictions on test set
    print("Making predictions...")
    y_pred_proba = stacking_model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold
    print("Finding optimal threshold...")
    thresholds = np.arange(0.2, 0.8, 0.05)
    best_f1 = 0
    best_threshold = 0.5  # Default
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Apply optimal threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Create test dataframe with predictions
    test_indices = np.arange(len(df))[len(y_train):]
    test_df = df.iloc[test_indices].copy()
    test_df["Predicted_Values"] = y_pred
    test_df["Prediction_Probability"] = y_pred_proba
    
    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n* Final Test Accuracy: {accuracy:.4f}")
    print(f"* F1 Score (Bad Responses): {f1:.4f}\n")
    print("* Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n* Confusion Matrix:")
    print(f"   True Negative: {cm[0][0]}, False Positive: {cm[0][1]}")
    print(f"   False Negative: {cm[1][0]}, True Positive: {cm[1][1]}")
    
    # Analyze beer preferences
    print("\n* Beer Preference Analysis")
    
    # Most common products
    print("\n* Most Common Products:")
    for col in beer_preference_columns:
        value_counts = df[col].value_counts().head(5)
        print(f"\n   {col}:")
        for product, count in value_counts.items():
            percentage = count / len(df) * 100
            print(f"     {product}: {count} responses ({percentage:.1f}%)")
    
    # Products by quality label
    print("\n* Products by Quality Label:")
    
    # Good responses (0)
    print("\n   Quality = 0 (Good):")
    for col in beer_preference_columns:
        value_counts = df[df["Quality_Flag_Binary"] == 0][col].value_counts().head(3)
        print(f"   - {col}:")
        for product, count in value_counts.items():
            percentage = count / len(df[df["Quality_Flag_Binary"] == 0]) * 100
            print(f"     {product}: {count} responses ({percentage:.1f}%)")
    
    # Bad responses (1)
    print("\n   Quality = 1 (Bad):")
    for col in beer_preference_columns:
        value_counts = df[df["Quality_Flag_Binary"] == 1][col].value_counts().head(3)
        print(f"   - {col}:")
        for product, count in value_counts.items():
            percentage = count / len(df[df["Quality_Flag_Binary"] == 1]) * 100
            print(f"     {product}: {count} responses ({percentage:.1f}%)")
    
    # Accuracy by product preference
    print("\n* Accuracy by Product Preference:")
    for col in beer_preference_columns:
        top_products = df[col].value_counts().head(5).index
        for product in top_products:
            product_indices = test_df[test_df[col] == product].index
            if len(product_indices) > 0:
                product_y_test = y_test[np.where(np.isin(test_indices, product_indices))]
                product_y_pred = y_pred[np.where(np.isin(test_indices, product_indices))]
                if len(product_y_test) > 0:
                    product_accuracy = accuracy_score(product_y_test, product_y_pred)
                    bad_responses_pct = np.mean(product_y_test) * 100
                    print(f"   {col}:")
                    print(f"     {product} (n={len(product_y_test)}): Accuracy={product_accuracy:.4f}, Bad Responses={bad_responses_pct:.1f}%")
    
    # Save model components
    model_file = 'stacking_ensemble_model.pkl'
    model_components = {
        'stacking_model': stacking_model,
        'scaler': scaler,
        'sbert_model': sbert_model,
        'threshold': best_threshold
    }
    with open(model_file, 'wb') as f:
        pickle.dump(model_components, f)
    print(f"\n* Model saved to '{model_file}'")
    
    # Save results to Excel
    output_file = "stacking_ensemble_results_with_beer_preferences.xlsx"
    test_df.to_excel(output_file, index=False)
    print(f"\n* Test Results saved to '{output_file}'**")

if __name__ == "__main__":
    xlsx_file = "Final Data File_Training.xlsx"
    sheet_name = "Data Set with Labels Text"
    main(xlsx_file, sheet_name)
