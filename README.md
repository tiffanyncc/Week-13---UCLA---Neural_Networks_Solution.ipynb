# UCLA Admission Prediction

## Project Scope

The world is developing rapidly, and continuously looking for the best knowledge and experience among people. This motivates people all around the world to stand out in their jobs and look for higher degrees that can help them in improving their skills and knowledge. As a result, the number of students applying for Master's programs has increased substantially.

The current admission dataset was created for the prediction of admissions into the University of California, Los Angeles (UCLA). It was built to help students in shortlisting universities based on their profiles. The predicted output gives them a fair idea about their chances of getting accepted.

## Aim

Build a classification model using Neural Networks to predict a student's chance of admission into UCLA.

## Specifics

- **Machine Learning task**: Classification model
- **Target variable**: Admit_Chance
- **Input variables**: Refer to data dictionary below
- **Success Criteria**: Accuracy of 90% and above

## Data Dictionary

The dataset contains several parameters which are considered important during the application for Masters Programs. The parameters included are:

- **GRE_Score**: (out of 340)
- **TOEFL_Score**: (out of 120)
- **University_Rating**: It indicates the Bachelor University ranking (out of 5)
- **SOP**: Statement of Purpose Strength (out of 5)
- **LOR**: Letter of Recommendation Strength (out of 5)
- **CGPA**: Student's Undergraduate GPA(out of 10)
- **Research**: Whether the student has Research Experience (either 0 or 1)
- **Admit_Chance**: (ranging from 0 to 1)

## Installation

1. Clone the repository:
    ```
    git clone <repository_url>
    ```

2. Navigate to the project directory:
    ```
    cd ucla_admission_prediction
    ```

3. Create a virtual environment:
    ```
    python -m venv venv
    ```

4. Activate the virtual environment:
    - On Windows:
      ```
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```
      source venv/bin/activate
      ```

5. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Running the Project

1. Ensure the data file `Admission.csv` is placed in the `data/raw` directory.

2. Run the main script:
    ```
    python main.py
    ```

3. The script will load the data, preprocess it, train a neural network model, evaluate it, and save the resulting plots in the `visualization/images` directory.

## Additional Information

- The project uses a feedforward neural network with 2 hidden layers.
- Hyperparameter tuning is performed using GridSearchCV.
- All generated plots are saved in the `visualization/images` directory.
