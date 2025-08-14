# rg_behavior_model
Implementation of the analysis and results shown in _Garza et al 2025_.


### Environment variables
In order for all the figure-generating scripts to get access to the right path and data, you will need to create a file 
named `.env` in the same directory as this README file.
Then open it and copy paste this template:
```angular2html
PATH_SAVE=<path_local>/results
PATH_DIR=<path_local>/data
```
Then substitute `<path_local>` with the actual path to the directory where the `data` folder is located.