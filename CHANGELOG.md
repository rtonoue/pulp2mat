# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.1.3] - 2022-08-21
### Changed
- Dictionary of LpVariable is no more required. Variable properties is now obtained from LpProblem.variables().

## [0.1.2] - 2022-08-17
### Added
- Maximization problem is now available.
- decode_solution() function to set scipy optimization result on LpVariable. 

