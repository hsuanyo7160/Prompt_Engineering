!pip install groq
import pandas as pd
import time
import os
from groq import Groq
from tqdm import tqdm
import random



# API Key
os.environ["GROQ_API_KEY"] = ""
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Model Selection
model_name = "deepseek-r1-distill-llama-70b"

######### Step 1: Generate the answer #########

# Prompt1
def generate_prompt1(question, options, subject):
    return f"""
There is a {subject.removeprefix("high_school_")} related question: {question} {options}
Based on a internal reasoning process, select the most correct answer and output only the letter of that option (without punctuation at the end)
"""

# Read 
df = pd.read_csv("/kaggle/input/gai-hw1/mmlu_submit.csv")

# Predict
results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row["input"]
    options = f"A: {row['A']}, B: {row['B']}, C: {row['C']}, D: {row['D']}"
    subject = row["task"]

    reasoning_output = "UNKNOWN"
    for attempt in range(5):
        try:
            prompt1 = generate_prompt1(question, options, subject)
            response1 = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt1}],
                    temperature=0,
                    max_tokens=10000  # 只回傳單一字母
            )
            reasoning_output = response1.choices[0].message.content.strip().upper()
            break
        except Exception as e:
            print(f" API Request Fail: {e}, {attempt + 1}th retrt")
            time.sleep(2 ** attempt + random.uniform(0, 1))
    print(" ", reasoning_output[-1])
    reasoning_output = reasoning_output[-1]
    results.append(reasoning_output)
    time.sleep(3)

# Save as CSV
result_df = pd.DataFrame({
    "ID": df["Unnamed: 0"],  #  ID
    "target": results  # Predicted Answer
})
result_csv_path = "/kaggle/input/gai949/result_temp.csv"
result_df.to_csv(result_csv_path, index=False)

######### Step 2: Generate the final answer #########

# Prompt2
def generate_prompt2(question, options, subject,ans):
    return f"""
There is a {subject.removeprefix("high_school_")} related question: {question} {options}.  
my answer is {ans}, carefully check whether this answer is correct and select the most correct answer
Output only the letter of that option (without punctuation at the end)
"""

# Read
df_ans = pd.read_csv("/kaggle/input/gai949/result_temp.csv")

# Predict
results = []
for index in tqdm(range(len(df)), desc="Processing Questions"):
    question = df.loc[index, "input"]
    options = f"A: {df.loc[index, 'A']}, B: {df.loc[index, 'B']}, C: {df.loc[index, 'C']}, D: {df.loc[index, 'D']}"
    subject = df.loc[index, "task"]
    ans = df_ans.loc[index, "target"].strip().upper()

    reasoning_output = "UNKNOWN"
    for attempt in range(5):
        try:
            prompt1 = generate_prompt2(question, options, subject, ans)
            response1 = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt1}],
                    temperature=0.1,
                    max_tokens=10000
            )
            reasoning_output = response1.choices[0].message.content.strip().upper()
            break
        except Exception as e:
            print(f" API Request ail: {e},  {attempt + 1} th retry")
            time.sleep(2 ** attempt + random.uniform(0, 1))

    if reasoning_output[-1] not in ["A", "B", "C", "D"]:
        print(reasoning_output)

    reasoning_output = reasoning_output[-1]
    print(" ",reasoning_output)
    results.append(reasoning_output)
    time.sleep(3)

# Save as CSV
result_df = pd.DataFrame({
    "ID": df["Unnamed: 0"],  #  ID
    "target": results  # Predicted Answer
})
result_csv_path = "final.csv"
result_df.to_csv(result_csv_path, index=False)