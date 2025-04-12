# CS 5100 Project 

A GitHub repository for an AI conversational agent that uses sentiment analysis to detect and mitigate biases in responses and responses with a bias score for responses.


# Clone the repository
git clone https://github.com/saahiwrites/CS5100_Project.git
cd CS5100_Project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


Core Features

Bias Detection through Sentiment Analysis: Detects potential biases in generated responses using advanced sentiment analysis techniques
Real-time Debiasing: Reframes responses to reduce detected biases while preserving the core message
Customizable Bias Categories: Easily configure which types of biases to detect and mitigate
Model Agnostic: Works with various LLM backends (OpenAI, Anthropic, open-source models)
Transparent Bias Metrics: Provides quantitative measures of bias reduction

# Usage Example
from src.agent import DeBiasedAgent

# Initialize the agent
agent = DeBiasedAgent(
    model="gpt-3.5-turbo",
    bias_categories=["gender", "race", "age"],
    sensitivity=0.7  # Adjust sensitivity of bias detection
)

# Generate a debiased response
response = agent.generate_response(
    user_input="What makes a good manager?",
    context={"domain": "workplace"}
)

print(response)
# The response will have reduced biases related to specified categories
