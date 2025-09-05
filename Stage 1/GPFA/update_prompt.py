import pandas as pd
import numpy as np
from GIFA.gpt_api import call_GPT
import time

class IntentPromptUpdater:
    """
    A class to update the intent discovery prompt based on evaluation metrics and error analysis.
    """

    def __init__(self, dynamic_prompt, metrics=None):
        """
        Initialize the updater with evaluation metrics and the current prompt.
        
        Parameters:
        - metrics (dict): A dictionary containing evaluation metrics with keys:
            "nca", "ari", "nmi", "ami", "fmi", "num_false_splits", "num_false_merges", and "interesting_examples".
        - dynamic_propmt (str): The current intent discovery prompt.
        """

        self.dynamic_prompt = dynamic_prompt
        self.metrics = metrics

        self.static_prompt = '''\n\n
        ### Instructions for Identifying Intents

        **Goal:** assign an intent to *every* new message.
        For each message, either reuse an intent from the existing intent list or create a new one if no suitable intent exists.

        
        Rules
        -----
        - Messages start with “assistant:” or “customer:”.
        - **Do NOT skip messages.**
        - Exactly **one** intent per line.
        - Use underscores, no capitals, no spaces.
        - Newly created intents must not duplicate an existing one.
        - Afterwards the new intents become part of the growing intent list.
        - **In this case {message_count} messages have to be assigned** in coupled intents.

        \n Return the output in EXACT format as specified below:
                                
        ### EXACT Output Format (only return the coupled intents list):

        Coupled intents:
        1. [intent_name]
        2. [intent_name]
        3. [intent_name]
        ...

        ### Starting Point
        Existing intents: {intent_list}

        ---
        ### Conversation for Analysis
        {trunc}
        '''

    def create_base_static_prompt(self):
        base_static_prompt = self.dynamic_prompt + self.static_prompt
        return base_static_prompt
    
    def create_meta_feedback_prompt(self, fast=False):
        """
        Generate a flexible meta prompt for refining the intent discovery process.
        
        Returns:
        - str: The meta prompt.
        """

        if not fast:
            global_metrics_text = f'''
                        ### Global Metrics

                        - **Neighborhood Clustering Accuracy (NCA):** {self.metrics['Clustering Metrics']["NCA"]:.4f}
                        - **Adjusted Mutual Information (AMI):** {self.metrics['Clustering Metrics']["AMI"]:.4f}

                        These metrics reflect the **local accuracy** (NCA) and the **structural similarity** (AMI) between the predicted clusters and the true intent labels. Low values in either indicate potential issues with the prompt’s ability to guide accurate or granular clustering.

                        ---

                        ### Intent Count Comparison

                        - **Predicted Number of Intents:** {self.metrics['Amounts']['Predicted Amount']}
                        - **Expected Number of Intents:** {self.metrics['Amounts']['True Amount']}

                        This comparison helps determine whether the clustering is **too fine-grained** (over-splitting) or **too coarse** (under-merging).

                        ---
                        '''
            global_metrics_explanation = f''' - Suggest adjustments to improve local grouping (NCA)
                                              - Recommend changes to better match the overall structure and granularity of the intents (AMI)
                                          '''
        else:
            global_metrics_text = ""
            global_metrics_explanation = ""

        meta_prompt = f'''
                        You are an advanced AI system assigned to improve an intent discovery prompt for clustering customer service messages.

                        Below is an evaluation summary of the current clustering performance:

                        ---

                        {global_metrics_text}

                        ### Error Examples

                        Each of the following pairs shows two messages, their true and predicted intents, and the type of error:

                        {self.metrics['Errors']["Interesting Examples"]}

                        ---

                        ### Task

                        Based on this evaluation, generate a **list of specific feedback** to improve the existing intent discovery prompt.

                        You may:
                        {global_metrics_explanation}
                        - Include **as many examples from the list above as necessary** to illustrate critical issues
                        - Propose ways to rephrase instructions, add disambiguation cues, or encourage more/less specificity

                        Avoid summarizing the metrics again. Focus on **feedback that would help rewrite the intent discovery prompt** to produce more accurate and better-structured clusters.
                        '''

        return meta_prompt.strip()

    def create_meta_prompt(self):
        """
        Generate the meta prompt string using the evaluation metrics.
        
        Returns:
        - str: The meta prompt.
        """
    
        meta_prompt = f'''
        You are an advanced AI system tasked with refining our intent discovery prompt for clustering customer service messages. Our evaluation of the current clustering approach has produced the following diagnostic results:

        ---

        ### 1. Global Clustering Metrics
        - Neighborhood Clustering Accuracy (NCA): {self.metrics['Clustering Metrics']["NCA"]:.4f}
        - Adjusted Mutual Information (AMI): {self.metrics['Clustering Metrics']["AMI"]:.4f}
        - Adjusted Rand Index (ARI): {self.metrics['Clustering Metrics']["ARI"]:.4f}
        - Fowlkes-Mallows Index (FMI): {self.metrics['Clustering Metrics']["FMI"]:.4f}

        These metrics indicate that the predicted clusters deviate significantly from the true intent structure.

        ---

        ### 2. Error Analysis
        - **False Splits:** {self.metrics['Errors']["False Splits"]}  
        *(Same true intent → assigned to different predicted clusters. Indicates over-fragmentation.)*
        - **False Merges:** {self.metrics['Errors']["False Merges"]}  
        *(Different true intents → assigned to the same predicted cluster. Indicates conflation of distinct intents.)*

        ---

        ### 3. Intent Count
        - **Predicted number of intents:** {self.metrics['Amounts']['Predicted Amount']}
        - **Expected (true) number of intents:** {self.metrics['Amounts']['True Amount']}

        **→ Interpretation:**  
        If the predicted number is **lower** than expected, the clustering is too coarse — encourage defining **more specific intents**.  
        If the predicted number is **higher**, the clustering is too fine — encourage defining **broader, more general intents**.

        ---

        ### Task

        Based on this analysis, **refine and improve the existing intent discovery prompt** to enhance clustering performance.

        You may **modify, remove, or extend** parts of the original prompt as needed.  
        You may also **build on top of it** by appending additional instructions or examples if the foundation is already solid.

        Your updated prompt should:

        1. **Adjust Granularity:**  
        Adapt the specificity of intent definitions based on current clustering behavior.

        2. **Improve Disambiguation Instructions:**  
        Emphasize that:
        - Messages with the **same true intent** should always be clustered together.
        - Messages with **different intents** must be clearly separated.

        3. **Enhance with Contextual Cues:**  
        Encourage using message content, customer/assistant roles, phrasing, and functional purpose to differentiate intents.

        4. **Incorporate Examples:**  
        You can include multiple examples (see below) to clarify distinctions between similar or merged/split intents.

        ---

        ### Output Format (IMPORTANT)

        Return **only** the updated prompt for intent discovery.  
        It may be a modified or extended version of the original — whichever is more appropriate.  
        Do **not** include any explanation, meta-analysis, or additional commentary.

        ### Error Examples

        Each tuple includes two messages, their true and predicted intents, and the error type:  
        {self.metrics['Errors']["Interesting Examples"]}

        ---

        '''
        return meta_prompt.strip()
    

    def create_feedback_updated_prompt(self, fast=False, temperature=0.7):

        feedback_prompt = self.create_meta_feedback_prompt(fast=fast)
        feedback_prompt = feedback_prompt + "\n Create feedback for the following prompt: \n" + self.dynamic_prompt
        feedback = call_GPT(feedback_prompt)

        updated_prompt = "Apply the following feedback to remake the initial prompt, return only the new prompt, this was the feedback: \n" + feedback + "THIS WAS THE Initial PROMPT: \n" + self.dynamic_prompt
        updated_dynamic_prompt = call_GPT(updated_prompt, temperature=temperature)

        updated_static_prompt = updated_dynamic_prompt + self.static_prompt

        return updated_static_prompt, updated_dynamic_prompt

    def create_updated_prompt(self, fast):
        """
        Create the updated intent discovery prompt by combining the meta prompt with the current prompt,
        then calling the GPT API to generate the revised prompt.
        
        Returns:
        - str: The updated intent discovery prompt.
        """
        meta_prompt = self.create_meta_feedback_prompt(fast=fast)
        full_prompt = meta_prompt + "\n This is the current prompt: \n" + self.dynamic_prompt + "\n RETURN ONLY THE NEW PROMPT"

        time.sleep(5)

        try:
            updated_dynamic_prompt = call_GPT(full_prompt)
        except Exception as e:
            print("Failed the OpenAI API:", e)
            time.sleep(61)
            updated_dynamic_prompt = call_GPT(full_prompt)

        updated_static_prompt = updated_dynamic_prompt + self.static_prompt

        return updated_static_prompt, updated_dynamic_prompt
    

