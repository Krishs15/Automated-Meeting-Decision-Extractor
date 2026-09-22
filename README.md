# Automated Meeting Decision Extractor

> A lightweight Python application that processes meeting transcripts, applies NLP preprocessing, and uses a trained decision-classification model to automatically extract key decisions and action items. 

Ideal for summarizing discussions and generating concise, actionable meeting minutes, this tool reduces the manual overhead of administrative meeting tasks.

---

## 📖 Overview

In fast-paced environments, tracking decisions and actionable tasks from meetings can be tedious and prone to human error. **Automated Meeting Decision Extractor** leverages Natural Language Processing (NLP) to read raw transcript data, clean it, and classify sentences into "decisions," "action items," or "general conversation." The result is a clean, structured summary of what was actually decided and who needs to do what.

## ✨ Key Features

*   **Robust NLP Preprocessing:** Automatically handles noise, filler words, and formatting inconsistencies common in raw meeting transcripts.
*   **Decision-Classification Model:** Employs a custom-trained machine learning model to accurately isolate sentences that contain commitments, action items, or final decisions.
*   **Lightweight & Fast:** Designed to run efficiently in standard Python environments without requiring heavy cloud infrastructure.
*   **Actionable Outputs:** Generates clean, concise meeting minutes that are immediately ready to be shared with stakeholders.

## ⚙️ Pipeline & Methodology

1.  **Transcript Ingestion:** Reads text files or standard transcript formats (e.g., from Zoom, Teams, or Google Meet).
2.  **Text Preprocessing:** Tokenization, stop-word removal, and dependency parsing to prepare the text for the model.
3.  **Classification:** The core model evaluates the intent of each sentence to extract high-value information.
4.  **Formatting:** Groups the extracted data into readable summaries.

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   `pip` package manager

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/Krishs15/Automated-Meeting-Decision-Extractor.git](https://github.com/Krishs15/Automated-Meeting-Decision-Extractor.git)
    cd Automated-Meeting-Decision-Extractor
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  *(Optional)* Download the pre-trained core NLP model, if relying on libraries like spaCy:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## 💻 Usage

Run the extractor on a sample transcript by executing the main script and passing your text file as an argument:

```bash
python extract_decisions.py --input data/sample_transcript.txt --output results/meeting_minutes.json
```

## Arguments:

--input: Path to the raw transcript file.

--output: (Optional) Path to save the structured JSON or markdown summary.

## 📊 Example Output
Input (Raw Transcript):

"Alice: So I think we should proceed with the AWS migration next month. Bob: I agree, let's finalize the budget by Friday. Charlie: Sounds good, I'll schedule a sync with the dev team tomorrow."

Output (Extracted Minutes):

```bash
{
  "Decisions": [
    "Proceed with the AWS migration next month."
  ],
  "Action Items": [
    "Finalize the budget by Friday. (Owner: Bob)",
    "Schedule a sync with the dev team tomorrow. (Owner: Charlie)"
  ]
}
```
## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📜 License
Distributed under the MIT License. See LICENSE for more information.
