import os
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
import re

from utils.LLM import ask_gpt

##clean all the text and add the colomun column_text that contains the text for the classifiers
vote={
    1:"terrible",
    2:"poor",
    3:"average",
    4:"good",
    5:"excellent",
}


def preprocess(df, field_name, column_text="cleaned_text", summary=False):
    # Add ID column as the first column
    new_df = pd.DataFrame()
    new_df["id"] = range(0, len(df))

    if not summary:
        if len(field_name) == 3: #for handle dataset that have only the column text
            title, review, rating = field_name


            df[review] = df[review].str.replace('The media could not be loaded.', '', regex=False).str.strip()

            new_df["review_text"] = df[title].apply(lambda x: x) +" "+df[review].apply(lambda x: x)

            new_df[column_text] = df[rating].apply(lambda x : vote[x])+" "+df[title].apply(lambda x: clean_text(x)) +" "+df[review].apply(lambda x: clean_text(x))

            #df = df.sample(frac=1)
        elif len(field_name) == 2:
            title, text = field_name
            new_df["review_text"] = df[title].apply(lambda x: x) + " " + df[text].apply(lambda x: x)

            new_df[column_text] =df[title].apply(lambda x: clean_text(x)) + " " + df[text].apply(lambda x: clean_text(x))
        else:
            text = field_name[0]
            new_df["review_text"] = df[text].apply(lambda x: x)
            new_df[column_text] = df[text].apply(lambda x: clean_text(x))

    else:

        new_df["review_text"] = df["summary"].apply(lambda x: x)
        new_df[column_text] = df["summary"].apply(lambda x: clean_text(x))



    return new_df


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ","
    text = re.sub(r"http\S+", "", text)  # Removing URLs
    # Remove words with less than 3 letters
    text = re.sub(r'\b\w{1,2}\b', '', text)
    # Remove extra spaces caused by the removal
    text = re.sub(r'\s+', ' ', text).strip()

    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)  # Removing html tags
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')  # Removing punctuations

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)  # Removing emojis
    words = word_tokenize(text.lower())
    words = [w.lower() for w in words if w.isalnum()]
    words = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)



def generate_summary(df, data_type, file_name):
    file_path = f"data/{data_type}/gpt_summary/{file_name}.csv"
    os.makedirs(f"data/{data_type}/gpt_summary/", exist_ok=True)
    prompt= """
    Summarize the following review in exactly 60 words, maintaining the first-person perspective and preserving the original tone.
    Use as many words from the review as possible while ensuring clarity and coherence.
    Retain the key points and overall sentiment of the text.
    
    Review: """

    if os.path.exists(file_path):
        summary = pd.read_csv(file_path)
    else:
        print()
        summary_list=[]
        for i,row in df.iterrows():
            review=row["review_text"]
            short_review=len(review.split()) < 80
            summary_list.append(review if short_review else ask_gpt(prompt + review))

            print(f"\r  Generating summary:{i}/{len(df)}",end="")
        summary=pd.DataFrame()
        summary['id']=range(len(summary_list))
        summary['summary']=summary_list
        summary.to_csv(file_path, index=False)
        print()

    return summary


def join_summary(df, data_type, file_name):
    df_summary = pd.read_csv(f"data/{data_type}/gpt_summary/{file_name}.csv")
    df_merged = pd.merge(df_summary, df, how="inner", on="id")
    return df_merged
