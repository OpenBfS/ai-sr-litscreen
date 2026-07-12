# AI-SR-LitScreen: LLM-assisted Title and Abstract Screening

This repository provides ready-to-use workflows for LLM-assisted title/abstract (TiAb) screening as described in a manuscript in submission.  
Two main workflows are implemented:

- **Open-source LLM workflow** (Fs-NR-L(os) in the paper)  
  - Notebook: `run-TiAbScreen_opensource.ipynb`
- **OpenAI LLM workflow** (Fs-M in the paper)  
  - Notebook: `run-TiAbScreen_openai.ipynb`  
  - Supports both **batch API** and **on‑demand API** usage.

## Getting started

1. **Clone this repository** and install the required Python dependencies ('requirements.txt'). Use a virtual environment preferably:

   ```bash
   git clone https://github.com/OpenBfS/ai-sr-litscreen.git
   cd ai-sr-litscreen
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare input files** by replacing the provided dummy files with your own:
   - `examples.txt` – few‑shot examples (recommended: 4 TiAb records with PECOS‑wise decisions and reasons; at least one fully eligible and one with all or most PECOS elements ineligible).
   - `pecos_criteria.yaml` – PECOS eligibility criteria for your systematic review or research question.
   - `abs-screen.xlsx` or `abs-screen.ris` – TiAb records to be screened.
   - `assistance_prompts.txt` provides prompts for extracting PECOS criteria into `pecos_criteria.yaml` and generating or annotating few-shot examples for `examples.txt`, helping prepare the files and text data required by the screening notebooks.

3. **Run a workflow**:
   - Copy `template.env` to `.env`, then set `OPENAI_API_KEY` and, if needed, `OLLAMA_BASE_URL`.
   - Use the committed `config.txt`; it supplies the exact model IDs used by the notebooks.
   - For open‑source models, open `run-TiAbScreen_opensource.ipynb`.
   - For OpenAI models, open `run-TiAbScreen_openai.ipynb` and choose batch or on‑demand API mode as described in the notebook. Make sure to enter your Open API key in the `.env` file.

4. **Helper functions**:
   - `functions_openai.py` and `functions_opensource.py` contain shared helper functions and are imported by the notebooks.

5. **Output**:
   - You can export the labelled TiAb as the original Excel or RIS files, with the workflow-generated labels in a "Label" column or "LB" field, respectively.

## License

Code is released under the MIT License. See `LICENSE` for details.