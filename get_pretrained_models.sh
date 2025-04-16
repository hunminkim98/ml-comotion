#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

cd src/comotion_demo/data

# Download and extract model checkpoints
wget https://ml-site.cdn-apple.com/models/comotion/demo_checkpoints.tar.gz
tar zxf demo_checkpoints.tar.gz
rm demo_checkpoints.tar.gz

cd ../../..
