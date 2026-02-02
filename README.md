# garza_et_al_2026
Implementation of the analysis and plots shown in _Garza et al 2026_.

### Project organization
Here is an overview to navigate the project
- `figures`: contains all scripts generating raw versions of the figures in the manuscript
- `model`: contains the implementation of the model, together with a lightweight version of a modeling framework under development in the Bahl Lab
- `utils` and `service`: contain useful constants, functions, and utility classes, used to run analysis and compute core quantities appearing in the figures
- `data`: empty directory, please put here the data once you downloaded them (see paragraph _Data_)
- `results`: empty directory where output figures will be stored 

### Environment variables
In order for all the figure-generating scripts to get access to the right path and data, you will need to create a file 
named `.env` in the same directory as this README file.
Then open it and copy paste this template:
```angular2html
PATH_DIR=<path_local>/data
PATH_SAVE=<path_local>/results
```
Then substitute `<path_local>` with the actual path to the directory where the `data` folder is located.

### Data
Please download the datasets from [here](https://kondata.uni-konstanz.de/radar/en/dataset/kn59u9atf99nejfb?token=gbuuWeDDsgdBHftQPkiJ) and populate your `data` directory inside the project. Keep the organization of the 
directory unchanged to have the project tree organized consistently to the paths called in the scripts.

### Dependencies
To install the core dependencies with conda:
- Open terminal
- Navigate to root directory of this project
- Run `conda env create --file=environment.yaml`
