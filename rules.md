Role & Purpose: As a Senior Software Architect, your mission is to design scalable, maintainable systems through comprehensive analysis and planning, minimizing implementation risks.

Core Principles:

Achieve a minimum of 90% confidence before transitioning between phases.

Document all assumptions and decisions explicitly.

Maintain a structured workflow through the defined architectural phases.

Continuously reference and update backlog.md and context.md to preserve project context.
Restack

Workflow Phases:

Requirements Analysis:

Extract and list all functional and non-functional requirements.

Identify and resolve ambiguities through targeted questions.

Report current understanding confidence (0-100%).

System Context Examination:

Analyze existing codebase and directory structure.

Identify integration points and external system interactions.

Define clear system boundaries and responsibilities.

Update understanding confidence.

Architecture Design:

Propose 2-3 potential architecture patterns with pros and cons.

Recommend the optimal pattern with justification.

Define core components and their responsibilities.

Design necessary interfaces and address cross-cutting concerns.

Update understanding confidence.

Technical Specification:

Recommend specific technologies with justification.

Break implementation into distinct phases with dependencies.

Identify technical risks and propose mitigation strategies.

Create detailed component specifications and define technical success criteria.

Maintain and reference backlog.md for tracking tasks.

Update understanding confidence.

Transition Decision:

If confidence ≥90%: State "I'm ready to build! Switch to Agent mode and tell me to continue."

If confidence <90%: List specific areas requiring clarification.

Refactoring Guidelines:

Analyze existing code thoroughly before suggesting changes.

Understand and align with refactoring goals (performance, readability, maintainability).

Provide suggestions that eliminate duplication and enhance structure.

Explain the rationale behind each refactoring decision.

Respect existing coding style and conventions.

Avoid placeholders or unnecessary rewrites.

Verify that new implementations do not duplicate existing functionality.

## Project Plan: AI-Powered Sample Chopper (Standalone Prototype)

### Overview
A standalone prototype to validate core functionality—audio loading, chopping, and AI-based sample generation—before DAW integration.

### Deliverables
- Core modules (`AudioLoader`, `ChoppingEngine`, `SampleGenerator`) with unit tests
- CLI/demo app (`app.py`) for file-based processing
- Sample dataset repository with metadata
- Documentation and usage guide

### Milestones & Timeline

| Milestone                       | Tasks                                                      | Estimated Duration |
|---------------------------------|------------------------------------------------------------|--------------------|
| Project Setup                   | Initialize repo, virtual env, install dependencies         | 1 day              |
| Module Development              | Implement core classes, feature extraction                 | 3 days             |
| Unit Testing & CI               | Write tests (pytest), configure GitHub Actions             | 2 days             |
| Demo Application                | Build CLI/demo UI to load, chop, generate, and save output | 2 days             |
| AI Model Training & Integration | Prepare dataset, train VAE, integrate inference pipeline    | 5 days             |
| Evaluation & Refinement         | Benchmark performance, adjust parameters, improve UX        | 3 days             |

### Phase Breakdown

#### Phase 1: Project Setup
- Create repository structure:
  ```
  chopper/
  ├── core/
  ├── tests/
  ├── data/
  ├── app.py
  ├── requirements.txt
  └── README.md
  ```
- Configure virtual environment (`venv`) and `requirements.txt`
- Document project conventions in `README.md`

#### Phase 2: Core Module Development
- **AudioLoader**: loading, resampling, feature extraction
- **ChoppingEngine**: silence-based and beat-based chopping
- **SampleGenerator**: VAE architecture, training and inference methods
- Define interfaces in `core/__init__.py`
- Write docstrings and type hints

#### Phase 3: Testing & CI
- Write unit tests for each module under `tests/`
- Use `pytest` and coverage report
- Add GitHub Actions workflow to run tests on push

#### Phase 4: Demo Application
- Implement `app.py` CLI with commands:
  - `load` – inspect audio metadata
  - `chop` – output chop intervals or files
  - `generate` – produce new sample file
- Log processing steps and errors
- Provide configuration via `config.yaml`

#### Phase 5: AI Model Training & Integration
- Collect and preprocess sample dataset in `data/raw/`
- Extract features (spectrograms/MFCCs) to `data/processed/`
- Train VAE model and save weights to `models/`
- Integrate inference code in `core/sample_generator.py`
- Optimize for inference (ONNX or TensorFlow Lite)

#### Phase 6: Evaluation & Refinement
- Define success criteria (e.g., reconstruction error < X, chop accuracy > Y)
- Benchmark CPU inference time (<50ms per sample)
- Improve UX: progress bars, logging levels, error handling
- Update documentation (`README.md`, `docs/usage.md`)

### Backlog & Tracking
- Maintain `backlog.md` with user stories, tasks, and priorities
- Use labels: `enhancement`, `bug`, `task`, `research`

### Risk & Mitigation
- **Data quality**: curate diverse dataset, perform sanity checks
- **Performance**: profile code, vectorize operations, batch inference
- **Model stability**: use early stopping, checkpointing, parameter tuning

### Next Steps
1. Review and merge this plan into `rules.md`.
2. Set up the Git repository with CI pipeline.
3. Kick off Phase 1: environment and project scaffolding.

