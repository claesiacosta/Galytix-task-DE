# Galytix-task-DE - Phrase Matching Pipeline

This Phrase Matching Pipeline is a tool that finds the closest match for a given input phrase from a list of pre-defined phrases. It utilizes pre-trained word embeddings to calculate the similarity between phrases.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/claesiacosta/Galytix-task-DE.git 
    ```

2. Navigate to the project directory:

    ```bash
    cd Galytix-task-DE
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    Alternatively
    ```bash
    python setup.py install
    ```


## Usage

1. Run the main script `main.py`:

    ```bash
    python main/main.py
    ```

2. Follow the prompts to input phrases or type 'exit' to quit.

3. The pipeline will find the closest match for each input phrase and display the result.

## Files

- `main.py`: Main script to run the phrase matching pipeline.
- `processor.py`: Module containing the Processor class for data handling and processing.
- `data/phrases.csv`: CSV file containing the list of pre-defined phrases.
- `data/GoogleNews-vectors-negative300.bin`: setup to be downloaded and extracted, in case error package error, add manually the file in this directory.
- `output/distances.csv`: Output CSV file containing the distances between input phrases and their closest matches.

## Requirements

- Python 3.x
- Required libraries (specified in `requirements.txt`)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
