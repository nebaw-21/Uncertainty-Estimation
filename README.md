# confident_ai: Uncertainty Estimation in Tabular Regression

This project demonstrates uncertainty estimation in tabular regression using MC Dropout, Deep Ensembles, and Temperature Scaling on the California Housing dataset. All code, results, and analysis are in a Jupyter notebook for easy exploration and reproducibility.

## Project Structure
- `uncertainty_estimation.ipynb`: Main notebook with all code, results, and plots.
- `models/`: Contains the MLP model definition.
- `methods/`: Contains MC Dropout, Deep Ensemble, and Temperature Scaling implementations.
- `results/`: Saved plots and figures.
- `report/`: Final report in Markdown format.
- `data/`: (Empty, as data is loaded from sklearn.)
- `step_by_step_guide.txt`: Detailed step-by-step instructions for reproducing the project.

## How to Run
1. **Clone the repository and open in VS Code.**
2. **Install requirements:**
   ```sh
   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn scipy
   ```
3. **Open `uncertainty_estimation.ipynb` in VS Code.**
4. **Run all cells sequentially.**
   - This will train models, compute uncertainty, generate plots, and display results.
5. **View results:**
   - Plots are saved in `results/`.
   - The final report is in `report/analysis.md`.

## Description
- Implements and compares MC Dropout and Deep Ensembles for uncertainty estimation in regression.
- Uses temperature scaling for post-hoc calibration of uncertainty intervals.
- Provides code, metrics, plots, and a detailed report for easy understanding and reproducibility.

## Requirements
- Python 3.8+
- VS Code with Python and Jupyter extensions recommended
- See `step_by_step_guide.txt` for full setup and workflow details

---

For more details, see the notebook and the step-by-step guide.
