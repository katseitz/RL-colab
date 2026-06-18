<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT HEADER -->
<div align="center">

<h2 align="center">RL In-Scanner Task</h2>

  <p align="center">
    A PsychoPy probabilistic reward / reversal-learning task for in-scanner fMRI data collection in the Nusslock Lab.
    <br />
    <a href="https://github.com/katseitz/RL-colab"><strong>Explore the repo »</strong></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#task-structure">Task Structure</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#running-a-session">Running a Session</a></li>
    <li><a href="#data-and-outputs">Data and Outputs</a></li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains the **reinforcement-learning (RL) task** used for reward-related fMRI data collection in the BD2 project and ACNL (Northwestern University). The task is a probabilistic reward/reversal-learning paradigm built in PsychoPy and run during scanning, with stimulus timing synchronised to the scanner trigger.

The experiment is authored in PsychoPy Builder (`RL_in-scanner.psyexp`); the committed `RL_in-scanner_lastrun.py` is the script PsychoPy generates from it. Edit the `.psyexp` in Builder rather than the generated `.py`, since the `.py` is overwritten on each compile.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Task Structure

A session runs in this order:

1. **Practice** — a short instructed practice block (left/right choice with feedback) so the participant is comfortable before scanning.
2. **Runs loop** — by default **3 runs** of **50 trials** each. Each run contains:
   - `get_ready` / `scanner_trigger` — waits for the scanner trigger (TR) so the task is locked to acquisition.
   - `trials` loop — per trial: `cue` → `cue_response` → `outcome` → `fixation`.
   - `run_feedback` and `leftover_time_break` — end-of-run summary and a timed break before the next run.

Trial-by-trial ISI/ITI jitters are read from `rl_reversal_jitters.csv`, indexed by run and trial. Reward contingencies reverse within the session (hence *reversal* learning), and per-run reward state is reset at the start of each run.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [PsychoPy](https://www.psychopy.org/) (developed/tested on v2025.1.1)
* Python 3
* [pandas](https://pandas.pydata.org/) (jitter/condition file handling)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* **PsychoPy** — the Standalone build is recommended so the Python environment and dependencies are bundled. Match the version above to avoid Builder-component API differences.
* A **monitor calibration** defined in PsychoPy's Monitor Center matching the stimulus PC / projector, with the correct screen selected (see [Troubleshooting](#troubleshooting)).
* The data files included in the repo (`rl_reversal_jitters.csv` and any condition files) present alongside the experiment.

### Installation

1. Clone the repo:
   ```sh
   git clone https://github.com/katseitz/RL-colab.git
   ```
2. Open `RL_in-scanner.psyexp` in **PsychoPy Builder**, or load it into the **PsychoPy Runner**.
3. (Optional, to avoid pushing to the base project) point the remote at your own fork:
   ```sh
   git remote set-url origin https://github.com/<your_username>/RL-colab.git
   git remote -v   # confirm the change
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
## Running a Session

1. Open the experiment in Builder/Runner and make sure the **Run** toggle is selected, **not Pilot** — pilot mode is for testing and changes display/saving behaviour.
2. Start the task and complete the **experiment info** dialog:
   - participant ID / session fields, and
   - `restart_from_run` — leave blank for a normal session. To resume mid-session after an aborted attempt, set it to **2** or **3**; the task then runs `num_runs = 4 - restart_from_run` runs (e.g. set to `2` to run runs 2 and 3).
3. Confirm the correct **screen/monitor** is selected before launching (the screen index can change between tasks on a shared stimulus PC).
4. The task waits for the **scanner trigger** at the start of each run, so start it before/at the acquisition and let it sync.

### Exiting early — important

If you need to stop the task between runs, **press the `Escape` key** to quit. Escape lets PsychoPy shut down gracefully and attempt to save. **Avoid the red stop button in the PsychoPy Runner**, which force-quits the underlying process and can prevent already-collected data from being written. As an additional safeguard, the task saves to disk at the end of every run (see below), so any *completed* run is preserved either way — but `Escape` remains the safer habit.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATA -->
## Data and Outputs

For each session PsychoPy writes to the `data/` directory:

* `*.csv` — trial-by-trial wide-format data (primary output).
* `*.psydat` — PsychoPy's pickled data object.
* `*.log` — a timestamped run log. Keep the logging level at `info` or higher so the log captures trial-level (`data`/`exp`) entries; this log is what allows a session to be reconstructed if a CSV is ever incomplete.
* `*_backup.csv` - the task is configured to write a `*_backup.csv`  data file at the **end of each run**, not only at the very end of the task. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- STRUCTURE -->
## Repository Structure

```
RL-colab/
├── RL_in-scanner.psyexp          # PsychoPy Builder source (edit this)
├── RL_in-scanner_lastrun.py      # Auto-generated script (do not hand-edit)
├── rl_reversal_jitters.csv       # Per-run/trial ISI & ITI jitters
├── data/                         # Session outputs (csv / psydat / log)
├── images/                       # Stimuli / assets
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

This is internal research code. If you have an improvement, please branch and open a pull request rather than committing to `main`, and coordinate with the maintainer before changes that affect timing, data format, or scanner synchronisation.

1. Create a feature branch (`git checkout -b feature/AmazingFeature`)
2. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
3. Push to the branch (`git push origin feature/AmazingFeature`)
4. Open a pull request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Katharina Seitz — katharinaseitz2029@u.northwestern.edu

Project Link: [https://github.com/katseitz/RL-colab](https://github.com/katseitz/RL-colab)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* The Nusslock Lab and the BD2 scanning / data-collection team
* [PsychoPy](https://www.psychopy.org/)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>