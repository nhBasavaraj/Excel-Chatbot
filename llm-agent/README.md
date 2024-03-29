# llm-agent Approach

## Overview

The `llm-agent` approach involves utilizing Jupyter notebooks to interact with the LLM (Language Model) agent for generating responses based on a set of questions. Multiple notebooks are provided, each tailored to a specific LLM model and Facebook dataset. Additionally, a notebook (`dummy-CSV_LLM-Agent_OpenAI_Orders-data.ipynb`) is available for testing with dummy Orders data. The sample responses generated by the OpenAI LLM agent are saved in `CSV_LLM-Agent-OpenAI_Facebook-response.pdf` PDF files.

## Files

1. **CSV_LLM-Agent-Cohere_Facebook-data.ipynb:**
   - Jupyter notebook using the Cohere Command model (No specific versions) with the Facebook dataset.

2. **CSV_LLM-Agent-Gemini-Pro_Facebook-data.ipynb:**
   - Jupyter notebook using the Gemini-Pro LLM model (model_name="gemini-pro") with the Facebook dataset.

3. **CSV_LLM-Agent-LLama-2-Facebook-data.ipynb:**
   - Jupyter notebooks using two LLama-2 LLM models with the Facebook dataset:
     - Model 1: "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
     - Model 2: "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"

4. **CSV_LLM-Agent-Mistral-8X7B-Facebook-data.ipynb:**
   - Jupyter notebook using the Mistral 8X7B LLM model ("mistralai/mixtral-8x7b-instruct-v0.1:2b56576fcfbe32fa0526897d8385dd3fb3d36ba6fd0dbe033c72886b81ade93e") with the Facebook dataset.

5. **CSV_LLM-Agent-OpenAI_Facebook.ipynb:**
   - Jupyter notebook using the OpenAI GPT-3.5-turbo LLM model with the Facebook dataset.
   
6. **dummy-CSV_LLM-Agent_OpenAI_Orders-data.ipynb:**
   - Jupyter notebook using the OpenAI GPT-3.5-turbo LLM model with dummy Orders dataset.

7. **CSV_LLM-Agent-OpenAI_Facebook-response.pdf:**
   - PDF file storing responses generated by the OpenAI GPT-3.5-turbo LLM agent in response to questions related to the Facebook dataset.

8. **dummy-CSV_LLM-Agent_OpenAI_Orders-data_response.pdf:**
   - PDF file storing responses generated by the OpenAI GPT-3.5-turbo LLM agent in response to questions related to the dummy Orders dataset.

## Note

**Note:** The Cohere and LLama-2 LLM models may currently demonstrate inconsistent behavior, occasionally yielding incorrect responses or encountering errors. These issues are acknowledged and are currently being addressed for future improvements.



## Usage

To use the `llm-agent` approach, follow these steps:

1. Open the respective Jupyter notebook based on the LLM model and dataset you want to use.

2. Execute the notebook cells to interact with the LLM agent and generate responses.

3. Refer to the sample responses generated by the OpenAI LLM agent for the LLM agent's responses to the set of questions.

4. Adjust the input data or set of questions as needed for your specific use case.

## Data Source

The set of questions for testing the LLM agents can be found in the `data-source` folder.
