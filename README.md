## Setup

Add untracked folders:
```
mkdir data outputs resources
```

If you don't already have a way to make a python 3.10 venv:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

Make a venv:
```
python3.10 -m venv .env
source .env/bin/activate
```

Install requirements:
```
cd lunarloc
pip install -r requirements.txt
```

If you get an OOM error building the orbslam bindings, try again with:
```
CMAKE_BUILD_PARALLEL_LEVEL=1 pip install -r requirements.txt
```

Download the resources for orbslam:
```
cd ..
./resources.sh
```

## Running

To run orbslam on a dataset and add VO:
```
python lunarloc/pgo/run_orbslam.py -t dataset_name.lac
```

To run PGO between ground truth and orbslam (with dummy loop closures):
```
python lunarloc/pgo/run_pgo.py -t dataset_name_w_orbslam.lac
```

Plotting just the ground truth trajectory for a .lac dataset:
```
python lunarloc/datasets/plot_lac.py -t dataset_name.lac
```

Converting a dataset to an mp4 for playback (uses FrontLeft camera):
```
python lunarloc/datasets/lac_to_mp4.py -t dataset_name.lac
```