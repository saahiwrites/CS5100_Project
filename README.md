# CS 5100 Project 

A GitHub repository for an AI conversational agent that uses sentiment analysis to detect and mitigate biases in responses and responses with a bias score for responses. 


Github Framework: 
debiased-ai-agent/
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── bias_datasets/
│   │   ├── gender_bias.csv
│   │   ├── racial_bias.csv
│   │   └── age_bias.csv
│   └── evaluation/
│       ├── test_cases.json
│       └── benchmark_results.csv
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── bias_detector/
│   │   ├── __init__.py
│   │   ├── sentiment_analyzer.py
│   │   └── bias_metrics.py
│   ├── debiasing/
│   │   ├── __init__.py
│   │   ├── reframing.py
│   │   └── counterfactual.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   └── llm_interface.py
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py
│       └── text_processing.py
├── config/
│   ├── agent_config.json
│   └── model_config.json
├── notebooks/
│   ├── bias_analysis.ipynb
│   ├── sentiment_exploration.ipynb
│   └── model_evaluation.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_bias_detection.py
│   └── test_debiasing.py
└── examples/
    ├── basic_agent_usage.py
    ├── custom_bias_detection.py
    └── api_integration.py


# Clone the repository
git clone https://github.com/saahiwrites/CS5100_Project.git
cd CS5100_Project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


