```
VULNCRYPT

VulnCrypt is a command-line utility designed to analyze C code and generate an n-dimensional
vulnerability prediction vector based on Common Weakness Enumeration (CWE) categories.
Leveraging recurrent neural networks (RNNs), VulnCrypt processes code to predict the
likelihood of various vulnerabilities being present. This project is heavily
inspired by Project Achilles, which provides similar functionality for Java code.
```

```
PURPOSE AND FUNCTIONALITY

- Vulnerability Analysis: Detects potential vulnerabilities in C code, focusing on
  specific CWE types such as buffer overflows (CWE-119), resource management errors
  (CWE-399), and null pointer dereferences (CWE-476).
- Machine Learning: Utilizes RNN models trained on labeled datasets to identify
  patterns associated with vulnerabilities.
- Code Tokenization and Encoding: Transforms raw C code into tokenized
  sequences suitable for input into machine learning models.
- Command-Line Interface: Offers a CLI for users to input C code files
  and receive vulnerability predictions.
```

```
HOW TO USE VULNCRYPT

PREREQUISITES

- Python 3.8 or higher
- Virtual Environment (recommended)
- Required Python Libraries: See requirements.txt
- Dataset: Juliet Test Suite for C/C++ (not included in the repository)
```

```
INSTALLATION

# Clone the Repository
git clone https://github.com/YOUR_USERNAME/VulnCrypt.git
cd VulnCrypt

# Set Up the Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Unix/Linux
venv\Scripts\activate     # On Windows

# Install Dependencies
pip install -r requirements.txt
```

```
DATA PREPARATION

# Download the Juliet Test Suite
# Obtain the Juliet Test Suite for C/C++ from the NIST website: https://samate.nist.gov/SARD/testsuite.php
# Extract the contents and place them in `venv/data/raw/`.

# Preprocess the Data
python src/data_processing/preprocess_data.py
python src/data_processing/build_vocab.py
python src/data_processing/encode_sequences.py
python src/data_processing/split_data.py
```

```
MODEL TRAINING

# Train the Model
python src/model/train.py

# Evaluate the Model
python src/model/evaluate.py
```

```
USING THE CLI

# Analyze a C Code File
python cli.py path/to/your/code.c

# View the Output
# The CLI will display a vulnerability prediction vector indicating the likelihood of specific CWE vulnerabilities.
```

```
ACKNOWLEDGMENTS

- Project Achilles: VulnCrypt was heavily inspired by Project Achilles.
  Project Achilles served as a reference and guide for developing VulnCrypt’s
  architecture, particularly in implementing the vulnerability prediction
  pipeline for programming languages.
- Juliet Test Suite: The dataset used for training and evaluation
  is based on the Juliet Test Suite provided by NIST.
```

```
PROJECT STRUCTURE

VulnCrypt/
├── models/                # Trained models and checkpoints
├── src/
│   ├── data_processing/   # Data preprocessing scripts
│   ├── model/             # Model architecture and training scripts
├── tests/                 # Unit tests
├── cli.py                 # Command-line interface script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── venv/                  # Virtual environment and data (excluded from version control)
```

```
TECHNICAL DETAILS

- Programming Language: Python 3.8+
- Deep Learning Framework: PyTorch
- Model Architecture: Long Short-Term Memory (LSTM) networks
- Data Processing:
  - Tokenization: Custom tokenizer for C code
  - Encoding: Sequences are encoded using an integer mapping
  from a built vocabulary
- Input: Tokenized and encoded sequences of C code
- Output: n-dimensional vulnerability prediction vector corresponding
  to selected CWE types
```

```
LICENSE

This project is licensed under the MIT License. See the LICENSE file for details.
```
