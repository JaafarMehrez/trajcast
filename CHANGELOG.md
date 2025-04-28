# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added .github/workflows/tests.yaml to handle unittesting on github.
- Added files for unittests which were missing in [1.0.0].
- Added this changelog.

### Fixed
- Fixed small issue in trajcast/data/wrappers/_lammps.py to not save the timestep which results in unittests failing with ase versions >= 3.25.0.

### Removed
- Removed travis.yml as unittest will be handled by .github/workflows/tests.yaml.

## [1.0.0] - 2025-04-01

### Added
- Initial code release. 