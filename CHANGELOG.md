# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.1.5] - 2024-11-30
### Changed
- Support Python >=3.9

## [0.1.4] - 2024-11-24
### Changed
- The virtual environment for this repository is managed by [uv](https://docs.astral.sh/uv/) from now.

## [0.1.3] - 2022-08-21
### Changed
- Dictionary of LpVariable is no more required. Variable properties is now obtained from LpProblem.variables().

## [0.1.2] - 2022-08-17
### Added
- Maximization problem is now available.
- decode_solution() function to set scipy optimization result on LpVariable. 

