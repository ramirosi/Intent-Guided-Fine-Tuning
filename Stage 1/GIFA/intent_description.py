import polars as pl
from GIFA.gpt_api import call_GPT
import time
import re

class DescriptionProcessor:
    def __init__(self):
        self.data_dict = {}  # Dictionary to store unique values and their filtered DataFrame
        self.filtered_df = pl.DataFrame()  # Stores the latest filtered DataFrame
        self.iteration_dict = {} # Dictionary with the new intents found in the current iteration of the intent generation

    def df_to_string(self, df):
        df = df.drop(["index", "count"])
        # Create a dict from the df
        result_dict = df.group_by("intent").agg(pl.col("messages").alias("messages")).to_dict(as_series=False)
        final_dict = dict(zip(result_dict["intent"], result_dict["messages"]))
        # Create a string from the dict
        dict_str = "\n".join(f"{key}: {values}" for key, values in final_dict.items())

        return dict_str
    
    def meta_description_prompt(self, intents_string):
        description_prompt = '''
                        You are analyzing a set of intents and their associated examples. Your task is to generate a **concise and general description** for each intent that accurately captures its meaning and scope. 

                        ### **Goals of This Task**
                        1. **Generate a general description for each intent** based on both its name and provided examples.
                        2. **Ensure clarity and consistency** so that intents are well-defined and distinct.
                        3. **Capture the underlying meaning** behind the examples while keeping descriptions broad enough for scalability.
                        4. **Avoid over-specificity**—descriptions should generalize beyond the specific examples given while maintaining intent accuracy.

                        ### **Instructions**
                        - **Use both the intent name and examples** to derive the description.
                        - **Ensure the description fully captures the intent’s purpose** without being too broad or too narrow.
                        - **Rephrase and generalize the examples into a single, structured definition**.
                        - **Write the descriptions in the exact required format** as shown below:

                        ---
                        ### **Required Output Format**

                        1. intent_name1: description
                        2. intent_name2: description
                        3. intent_name3: description
                        ...

                        Ensure that:
                        - The **intent names are kept unchanged**.
                        - The **descriptions are clear and concise**.
                        - The **format strictly follows the numbered list**.

                        ### Intents and their examples:
                            {new_intents_string}
        
                        '''.format(new_intents_string = intents_string)
        return description_prompt
    
    def parse_intent_descriptions(self, text: str) -> dict:
        """
        Parses a multi-line string to extract intent descriptions.
        """
        intent_dict = {}

        for line in text.split("\n"):
            match = re.match(r"^\d+\.\s*([A-Za-z0-9_'\- ]+):\s*(.+)", line)
            if match:
                intent_name = match.group(1).strip()
                description = match.group(2).strip()
                intent_dict[intent_name] = description

        return intent_dict
    
    def update(self, df: pl.DataFrame, column: str):
        """Filters the DataFrame for values not yet in the dictionary, but does not update the dictionary."""
        unique_values = set(df[column].unique().to_list())  # Get unique values
        new_values = unique_values - set(self.data_dict.keys())  # Find only new values

        if not new_values:
            self.filtered_df = pl.DataFrame()  # Reset to empty DataFrame if nothing new
            return new_values, 0
        else:
            self.filtered_df = df.filter(pl.col(column).is_in(new_values))  # Filter for new values only

        self.filtered_df = self.filtered_df.filter(~pl.col("intent").str.contains(","))

        if len(self.filtered_df) == 0:
            return new_values, 0

        # Filter for new values only 

        new_intents_string = self.df_to_string(self.filtered_df)
        intents_prompt = self.meta_description_prompt(new_intents_string)

        # time.sleep(5)

        try:
            result, token_count = call_GPT(intents_prompt, return_token_count=True)
        except Exception as e:
            print("Failed the OpenAI API:", e)
            time.sleep(61)
            result, token_count = call_GPT(intents_prompt, return_token_count=True)

        new_intent_dict = self.parse_intent_descriptions(result)
        self.data_dict = self.data_dict | new_intent_dict

        # print(f"The new intent dict had {len(new_intent_dict)} new intents resulting in {len(self.data_dict)}")

        return new_values, token_count