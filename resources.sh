#!/usr/bin/env bash
set -e

mkdir -p resources
curl -L https://github.com/Robaire/MAPLE/raw/main/resources/ORBvoc.txt.tar.gz -o resources/ORBvoc.txt.tar.gz
curl -L https://github.com/Robaire/MAPLE/raw/main/resources/orbslam_config.yaml -o resources/orbslam_config.yaml
echo "*" > resources/.gitignore

tar -xzf resources/ORBvoc.txt.tar.gz -C resources
