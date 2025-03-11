import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field
import json
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


# Define Pydantic model for structured JSON output
class QualityAssessment(BaseModel):
    score: float = Field(
        ge=0,
        le=1,
        description="Quality score between 0 and 1. Should be between 0 and 1!",
    )
    reason: str = Field(
        max_length=1000,
        description="Short reason for the score. Should be under 100 characters!",
    )


# Load and preprocess the TEST dataset
df_test = pd.read_excel("./Final Data File_Test.xlsx")  # Changed to test dataset

# Define mappings for Q6 and Q7 options
q6_options = [
    "Beer",
    "Flavored/Mixed Beer",
    "Non-Alcoholic Beers",
    "Hard Ciders",
    "Hard Kombucha",
    "Wine",
    "Hard Beverage",
    "Distilled Spirits",
]

q7_options = [
    "Domestic Light Beer",
    "Domestic Regular Beer",
    "Low Calorie Carb Beer",
    "Domestic Craft Imported Mexican Beer",
    "Other Imported Beer",
    "Non-Alcoholic Beer",
]

# Get column names for Q6 and Q7
q6_cols = [
    "Q6 Which of the following types of alcoholic beverages have you consumed in the past 4 weeks?\n(Alcohol Category)"
]
q6_cols.extend([f"Unnamed: {i}" for i in range(8, 8 + len(q6_options) - 1)])

q7_cols = [
    "Q7. Which of the following beer types of have you consumed in the past 4 weeks? \n(Beer Category )"
]
q7_cols.extend([f"Unnamed: {i}" for i in range(16, 16 + len(q7_options) - 1)])


# Functions to get "Yes" responses for Q6 and Q7
def get_alcohol_categories(row):
    categories = []
    if isinstance(row[q6_cols[0]], str) and row[q6_cols[0]].strip().lower() == "yes":
        categories.append(q6_options[0])
    for i in range(1, len(q6_options)):
        if (
            i < len(q6_cols)
            and isinstance(row[q6_cols[i]], str)
            and row[q6_cols[i]].strip().lower() == "yes"
        ):
            categories.append(q6_options[i])
    return categories


def get_beer_categories(row):
    categories = []
    if isinstance(row[q7_cols[0]], str) and row[q7_cols[0]].strip().lower() == "yes":
        categories.append(q7_options[0])
    for i in range(1, len(q7_options)):
        if (
            i < len(q7_cols)
            and isinstance(row[q7_cols[i]], str)
            and row[q7_cols[i]].strip().lower() == "yes"
        ):
            categories.append(q7_options[i])
    return categories


# Apply functions to create new columns
df_test["Q6_Alcohol_Categories_Consumed_in_Past_4_Weeks_LIST"] = df_test.apply(
    get_alcohol_categories, axis=1
)
df_test["Q7_Beer_Categories_Consumed_in_Past_4_Weeks_LIST"] = df_test.apply(
    get_beer_categories, axis=1
)
df_test = df_test.iloc[1:]  # Drop the first row as in the original code

# Define question mappings
question_mappings = {
    "Q1": {
        "description": "Current age",
        "column_name": "Q1. What is your current age? \n(Age)",
    },
    "Q2": {
        "description": "Gender",
        "column_name": "Q2. What is your gender? \n(Gender)",
    },
    "Q3": {
        "description": "Area/community type (Urban/Rural)",
        "column_name": "Q3. Which of the following best describes the area or community in which you live? \n(Urban/Rural)",
    },
    "Q4": {
        "description": "Household income",
        "column_name": "Q4.  Please indicate the answer that includes your entire household income in (previous year) before taxes. \n(Income)",
    },
    "Q6": {
        "description": "Alcoholic beverages consumed in past 4 weeks",
        "column_name": "Q6_Alcohol_Categories_Consumed_in_Past_4_Weeks_LIST",
    },
    "Q7": {
        "description": "Beer types consumed in past 4 weeks",
        "column_name": "Q7_Beer_Categories_Consumed_in_Past_4_Weeks_LIST",
    },
    "Q9": {
        "description": "Product relevance",
        "column_name": "Q9. How relevant would you say the shown product is to you based on what you saw and read?\n(Concept Relevance)",
    },
    "Q10": {
        "description": "Product appeal",
        "column_name": "Q10. How appealing or unappealing is the shown product  to you?\n(Concept Appeal)",
    },
    "Q11": {
        "description": "Product differentiation",
        "column_name": "Q11. How different do you think the shown product is from other beers currently available for purchase?\n(Concept Differentiation)",
    },
    "Q12": {
        "description": "Product believability",
        "column_name": "Q12. Thinking about the shown product, which option describes how believable or unbelievable you feel the description and statements made about it are?\n(Concept Beleivability)",
    },
    "Q13": {
        "description": "Price expectation",
        "column_name": "Q13. How does the price fit with what you’d expect the shown to cost?\n(Concept_Price)",
    },
    "Q14": {
        "description": "Purchase intent",
        "column_name": "Q14. Which statement below best describes how likely you would be to buy shown product if it were available at your local stores?\n(Concept_Purchase Intent)",
    },
    "Q15": {
        "description": "Expected drinking frequency",
        "column_name": "Q15. If the shwon product was available to you, how often would you expect yourself to drink at least one of these products?\n(Concept_Drinking Frequency)",
    },
    "Q16A": {
        "description": "Most liked aspects",
        "column_name": "Q16A. What is the most important thing you LIKE about the shown concept}?     This can include anything you would want kept for sure or aspects that might drive you to buy or try it…       Please type a detailed response in the space below",
    },
    "Q16B": {
        "description": "Most disliked aspects",
        "column_name": "Q16B. What is the most important thing you DISLIKE about the shown concept}?    This can include general concerns, annoyances, or any aspects of the product that need fixed for this to be more appealing to you...     Please type a detailed response in the space below.",
    },
    "Q17": {
        "description": "Effect on other beverage purchases",
        "column_name": "Q17. We would like to know what effect this new product might have on the other beverages you buy. If it were available, would the shown product…? \n(Concept_Replacement Product)",
    },

    "Q18_1": {
        "description": "Specific Alcohol name product replacement name 1 which is an openended field not a multiple choice",
        "column_name": "Q18_1 What specific product that you are currently using would the shown product replace?\n Please type in ONE specific brand or product per space provided.",
    },
    "Q18_2": {
        "description": "Specific Alcohol name product replacement name 2 which is an openended field not a multiple choice",
        "column_name": "Q18_2 What specific product that you are currently using would the shown concept replace?\n Please type in ONE specific brand or product per space provided.",
    },
    "Q18_3": {
        "description": "Specific Alcohol name product replacement name 3 which is an openended field not a multiple choice",
        "column_name": "Q18_3 What specific product that you are currently using would the shown concept replace?\n Please type in ONE specific brand or product per space provided.",
    },
}

# Load training data to get examples (assuming training data is still needed for examples)
df_train = pd.read_excel("./Final Data File_Training.xlsx")
df_train["Q6_Alcohol_Categories_Consumed_in_Past_4_Weeks_LIST"] = df_train.apply(
    get_alcohol_categories, axis=1
)
df_train["Q7_Beer_Categories_Consumed_in_Past_4_Weeks_LIST"] = df_train.apply(
    get_beer_categories, axis=1
)
df_train = df_train.iloc[1:]
good_examples = df_train[df_train["OE_Quality_Flag"] == 0].sample(20, random_state=42)
bad_examples = df_train[df_train["OE_Quality_Flag"] == 1].sample(20, random_state=42)


# Enhanced prompt generation function with improved instructions
def generate_gemini_prompt(row, question_mappings, good_examples, bad_examples):
    profile_sections = []
    for q_key, q_info in question_mappings.items():
        value = row.get(q_info["column_name"], "[No data]")
        profile_sections.append(f"- {q_info['description']}: {value}")
    profile = "\n".join(profile_sections)

    examples = "## EXAMPLES FROM THE TRAINING DATASET (Assume a total of 20 examples provided, including both valid and invalid responses)\n\n"
    for _, ex_row in good_examples.iterrows():
        examples += "### GOOD QUALITY RESPONSE (Quality Flag: 0)\n"
        example_profile = [
            f"- {q_info['description']}: {ex_row.get(q_info['column_name'], '[No data]')}"
            for q_key, q_info in question_mappings.items()
        ]
        examples += "\n".join(example_profile) + "\n\n"
    for _, ex_row in bad_examples.iterrows():
        examples += "### BAD QUALITY RESPONSE (Quality Flag: 1)\n"
        example_profile = [
            f"- {q_info['description']}: {ex_row.get(q_info['column_name'], '[No data]')}"
            for q_key, q_info in question_mappings.items()
        ]
        examples += "\n".join(example_profile) + "\n\n"
        additional_ids = ["2262", "912", "503", "232", "1478", "1639", "908"]
        additional_examples = df_train[df_train["Unique ID"].astype(str).isin(additional_ids)]
        for _, add_row in additional_examples.iterrows():
            examples += "### EXTRA EXAMPLE\n"
            for q_key, q_info in question_mappings.items():
                examples += f"- {q_info['description']}: {add_row.get(q_info['column_name'], '[No data]')}\n"
            examples += "\n"
        examples += "### EXTRA EXAMPLE\n"
        for q_key, q_info in question_mappings.items():
            examples += f"- {q_info['description']}: {additional_examples.get(q_info['column_name'], '[No data]')}\n"
        examples += "\n"

    prompt = f"""
# SURVEY RESPONSE QUALITY EVALUATION

You are an AI assistant tasked with assessing the quality of survey responses for a beer survey. Your job is to determine if a response is acceptable or not by checking if it is on-topic, relevant, and sufficiently thoughtful. Pay special attention to open-ended responses about "Most liked aspects" and "Most disliked aspects". Flag responses that are completely off-topic, gibberish, contain profanity, or are overly repetitive (e.g., the same response for everything). However, do not be overly strict: if a response is acceptable even if somewhat lazy, or if it addresses the product and beer-related questions properly, mark it as valid.

Learn from the 20 examples and more provided below (10 acceptable and 10 unacceptable, as illustrated and rest as exception and all) to guide your evaluation. Accept responses that are on-topic and address the product and beers even if they seem a bit lazy.

LEARN FROM THE EXAMPLES ON WHAT TO OUTPUT FOR THE SCORE!!!! DO NOT MAKE IT TOO STRICT. IF THE RESPONSE IS LAZY IT IS ACCEPTABLE AND FINE! FINESH RESPONSE IS ALSO GOOD! ALSO ANY RESPONSE THAT IS TALKING ABOUT BEER EVEN WITH SPELLING MISTAKES IS MOSTY GOOD. DO NOT MISTAKE LAZY RESPONSES FOR BAD RESPONSES!
Also, if the response is talking about the product and beer, even if it is not perfect, it is still good.
If someone is answering yes to all questions, or says he doesn't drink beer and still says he have drank 10 beers in the past 4 weeks, that would be unacceptable.

## EXAMPLES FROM THE TRAINING DATASET
{examples}

## RESPONDENT PROFILE TO EVALUATE
{profile}

## EVALUATION INSTRUCTIONS
1. **Analyze the Responses**: Focus especially on open-ended answers like "Most liked aspects" and "Most disliked aspects".
2. **Check for Quality Issues**:
   - **Gibberish**: Random characters or nonsensical text.
   - **Profanity**: Inappropriate or offensive language.
   - **Off-Topic**: Answers unrelated to the survey questions.
   - **Repetition**: Excessive repetition or identical responses.
   - **Thoughtfulness**: Lack of meaningful content.
3. **Assign a Score**: Provide a score between 0 and 1:
   - 0 = High quality (clear, relevant, thoughtful).
   - 1 = Low quality (significant issues present).
4. **Provide a Reason**: In 1-2 sentences (under 150 characters) explain your score.



## OUTPUT FORMAT
Return your evaluation as a JSON object with this exact structure:
```json
{{
    "score": <float between 0 and 1>,
    "reason": "<your short reasoning>"
}}
```

### Examples of Expected Output
- High Quality:
```json
{{
    "score": 0.1,
    "reason": "Detailed and relevant feedback provided."
}}
```
- Low Quality:
```json
{{
    "score": 0.9,
    "reason": "Response contains gibberish and profanity."
}}
```

## YOUR TASK
Evaluate the respondent's profile above using the examples and instructions provided. Mark acceptable responses as valid, even if they appear a bit lazy, as long as they are on-topic and address the beer product appropriately.
"""
    return prompt


# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.0-flash-001")

# Add columns for predictions if not already present
for col in ["Predicted_Flag", "Score", "Reason"]:
    if col not in df_test.columns:
        df_test[col] = None

# Load existing results if available to resume processing
output_file = "predicted_quality_flags_test.csv"
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    processed_ids = set(df_existing["Unique ID"].dropna().astype(str))
    print(f"Resuming from {len(processed_ids)} previously processed rows.")
else:
    processed_ids = set()
    df_existing = pd.DataFrame(columns=df_test.columns)
    df_existing.to_csv(output_file, index=False)

# Evaluate model performance on a sample of training data
print("\nEvaluating model performance on training data sample...")

# Take 15 samples with OE_Quality_Flag = 0 and 15 with OE_Quality_Flag = 1
train_sample_0 = df_train[df_train["OE_Quality_Flag"] == 0].sample(15)
train_sample_1 = df_train[df_train["OE_Quality_Flag"] == 1].sample(15)
df_train_sample = pd.concat([train_sample_0, train_sample_1])

# Add columns for predictions if not already present
for col in ["Predicted_Flag", "Score", "Reason"]:
    if col not in df_train_sample.columns:
        df_train_sample[col] = None

# Process the sample
for index, row in tqdm(
    df_train_sample.iterrows(),
    total=len(df_train_sample),
    desc="Processing training sample",
):
    prompt = generate_gemini_prompt(row, question_mappings, good_examples, bad_examples)
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                # temperature=0.1,
            ),
        )
        json_response = json.loads(response.text)
        assessment = QualityAssessment(**json_response)
        score = assessment.score
        reason = assessment.reason
        predicted_flag = 1 if score >= 0.3 else 0  # Lowered threshold from 0.5 to 0.3
        df_train_sample.at[index, "Predicted_Flag"] = predicted_flag
        df_train_sample.at[index, "Score"] = score
        df_train_sample.at[index, "Reason"] = reason
    except Exception as e:
        print(
            f"Error processing training sample row {index} (Unique ID: {row['Unique ID']}): {e}"
        )
        df_train_sample.at[index, "Predicted_Flag"] = None
        df_train_sample.at[index, "Score"] = None
        df_train_sample.at[index, "Reason"] = f"Error: {str(e)}"

# Save training sample results
train_output_file = "train_sample_quality_flags.xlsx"
df_train_sample.to_excel(train_output_file, index=False)

# Calculate and display performance metrics
actual_flags = df_train_sample["OE_Quality_Flag"].astype(int)
predicted_flags = (
    df_train_sample["Predicted_Flag"].fillna(-1).astype(int)
)  # Treat errors as mismatches
correct_predictions = (actual_flags == predicted_flags).sum()
total_valid = (predicted_flags != -1).sum()  # Exclude errors from accuracy calculation
accuracy = correct_predictions / total_valid if total_valid > 0 else 0

print(f"\nTraining Sample Results:")
print(f"Total Samples: {len(df_train_sample)}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Valid Predictions (excluding errors): {total_valid}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Results saved to '{train_output_file}'.")

# Process each row with progress bar
for index, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Processing rows"):
    unique_id = str(row["Unique ID"])
    if unique_id in processed_ids:
        continue  # Skip already processed rows

    prompt = generate_gemini_prompt(row, question_mappings, good_examples, bad_examples)
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
        )
        json_response = json.loads(response.text)
        assessment = QualityAssessment(**json_response)
        score = assessment.score
        reason = assessment.reason
        predicted_flag = 1 if score >= 0.5 else 0  # Threshold at 0.5

        # Update DataFrame
        df_test.at[index, "Predicted_Flag"] = predicted_flag
        df_test.at[index, "Score"] = score
        df_test.at[index, "Reason"] = reason
    except Exception as e:
        print(f"Error processing row {index} (Unique ID: {unique_id}): {e}")
        df_test.at[index, "Predicted_Flag"] = None
        df_test.at[index, "Score"] = None
        df_test.at[index, "Reason"] = f"Error: {str(e)}"

    # Append the current row to the output CSV - THIS IS THE FIX
    df_row = df_test.loc[[index]]  # Use .loc instead of .iloc
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        df_row.to_csv(output_file, mode="a", header=False, index=False)
    else:
        df_row.to_csv(output_file, mode="w", header=True, index=False)

    # Update processed_ids
    processed_ids.add(unique_id)
