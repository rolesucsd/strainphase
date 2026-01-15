# Installation Guide

Complete installation instructions for Strainphase on various platforms.

## Table of Contents

- [Quick Install](#quick-install)
- [Detailed Installation](#detailed-installation)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Dependencies](#dependencies)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Updating](#updating)
- [Uninstalling](#uninstalling)

## Quick Install

### Using pip (Recommended)

```bash
# Basic installation
pip install strainphase

# Full installation (includes pysam for BAM/VCF I/O)
pip install strainphase[full]

# Development installation
pip install strainphase[dev]

# All dependencies
pip install strainphase[all]
```

### Using conda

```bash
# Create conda environment
conda create -n strainphase python=3.11
conda activate strainphase

# Install from PyPI
pip install strainphase[full]
```

## Detailed Installation

### Prerequisites

**System requirements:**
- Operating System: Linux, macOS, or Windows (WSL2 recommended)
- Python 3.11+ (recommended: 3.13)
- 4 GB RAM minimum (8+ GB recommended for large datasets)
- 1 GB disk space

**External tools (optional but recommended):**
- `minimap2`: For read alignment
- `samtools`: For BAM manipulation
- `bcftools`: For VCF operations
- `Clair3`: For variant calling from PacBio HiFi reads

### Step-by-Step Installation

#### 1. Set Up Python Environment

**Using venv (standard library):**
```bash
# Create virtual environment
python3 -m venv strainphase-env

# Activate environment
source strainphase-env/bin/activate  # Linux/macOS
# OR
strainphase-env\Scripts\activate     # Windows
```

**Using conda:**
```bash
# Create environment with specific Python version
conda create -n strainphase python=3.11

# Activate environment
conda activate strainphase
```

#### 2. Install Strainphase

**Option A: Install from PyPI (stable release)**
```bash
pip install strainphase[full]
```

**Option B: Install from source (latest development version)**
```bash
# Clone repository
git clone https://github.com/rolesucsd/strainphase.git
cd strainphase

# Install in development mode
pip install -e ".[all]"

# Install pre-commit hooks (optional)
pre-commit install
```

**Option C: Install from GitHub release**
```bash
pip install https://github.com/rolesucsd/strainphase/archive/v0.1.0.tar.gz
```

#### 3. Verify Installation

```bash
# Check version
strainphase version

# Run test suite
strainphase test

# View help
strainphase --help
```

## Platform-Specific Instructions

### Ubuntu/Debian Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Install build essentials (needed for some dependencies)
sudo apt-get install -y build-essential python3-dev

# Optional: Install external tools
sudo apt-get install -y samtools bcftools minimap2

# Install Strainphase
pip3 install strainphase[full]
```

### CentOS/RHEL/Fedora

```bash
# Install system dependencies
sudo yum install -y python3 python3-pip python3-devel gcc

# Install Strainphase
pip3 install strainphase[full]
```

### macOS

**Using Homebrew:**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Optional: Install external tools
brew install samtools bcftools minimap2

# Install Strainphase
pip3 install strainphase[full]
```

**Using MacPorts:**
```bash
# Install Python
sudo port install python311

# Install Strainphase
pip3.11 install strainphase[full]
```

### Windows (WSL2)

Windows Subsystem for Linux 2 (WSL2) is recommended for Windows users:

```bash
# Install WSL2 (PowerShell as Administrator)
wsl --install -d Ubuntu

# Launch Ubuntu and update
sudo apt update && sudo apt upgrade

# Follow Ubuntu instructions above
```

**Native Windows (not recommended):**
```powershell
# Install Python from python.org
# Then in PowerShell:
pip install strainphase

# Note: pysam may have issues on Windows
# Use WSL2 for best compatibility
```

### HPC/Cluster Environments

**Using environment modules:**
```bash
# Load Python module
module load python/3.11

# Create virtual environment in your home directory
python -m venv ~/.venvs/strainphase
source ~/.venvs/strainphase/bin/activate

# Install
pip install strainphase[full]
```

**Using conda on HPC:**
```bash
# Load conda module
module load miniconda3

# Create environment
conda create -n strainphase python=3.11
conda activate strainphase

# Install
pip install strainphase[full]
```

**Batch job example (SLURM):**
```bash
#!/bin/bash
#SBATCH --job-name=strainphase
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

# Load environment
source ~/.venvs/strainphase/bin/activate

# Run analysis
strainphase longitudinal \
  --samples T1,T2,T3 \
  --bams data/{sample}.bam \
  --vcfs data/{sample}.vcf.gz \
  --reference ref.fasta \
  --output-dir results/
```

## Dependencies

### Core Dependencies (automatic)

These are installed automatically with Strainphase:

- `numpy >= 1.20`: Numerical computations
- `scipy >= 1.7`: Statistical functions
- `networkx >= 2.6`: Graph algorithms

### Optional Dependencies

**Full installation (`pip install strainphase[full]`):**
- `pysam >= 0.19`: BAM/VCF I/O
- `python-louvain >= 0.16`: Community detection (better than connected components)
- `pandas >= 1.3`: Data manipulation

**Development (`pip install strainphase[dev]`):**
- `pytest >= 7.0`: Testing framework
- `pytest-cov >= 4.0`: Code coverage
- `black >= 23.0`: Code formatting
- `ruff >= 0.1`: Linting

**Documentation (`pip install strainphase[docs]`):**
- `sphinx >= 6.0`: Documentation generation
- `sphinx-rtd-theme >= 1.0`: ReadTheDocs theme

### External Tools

Not installed by pip - must be installed separately:

**minimap2:**
```bash
# Ubuntu/Debian
sudo apt-get install minimap2

# macOS
brew install minimap2

# From source
git clone https://github.com/lh3/minimap2
cd minimap2 && make
```

**samtools:**
```bash
# Ubuntu/Debian
sudo apt-get install samtools

# macOS
brew install samtools

# conda
conda install -c bioconda samtools
```

**Clair3:**
```bash
# Docker (recommended)
docker pull hkubal/clair3:latest

# conda
conda install -c bioconda clair3

# From source
git clone https://github.com/HKU-BAL/Clair3.git
cd Clair3 && ./install.sh
```

## Verification

### Basic Verification

```bash
# Check installation
python -c "import strainphase; print(strainphase.__version__)"

# Verify CLI works
strainphase --help

# Check all subcommands
strainphase run --help
strainphase longitudinal --help
strainphase test --help
strainphase sweep --help
```

### Comprehensive Testing

```bash
# Run full test suite
strainphase test -v

# Run with coverage
pytest --cov=strainphase src/strainphase/tests/

# Run parameter sweep (takes a few minutes)
strainphase sweep --quick
```

### Verify Dependencies

```bash
# Check Python version
python --version  # Should be 3.9+

# Check installed packages
pip list | grep -E "(strainphase|numpy|scipy|networkx|pysam)"

# Check optional tools
minimap2 --version
samtools --version
bcftools --version
```

## Troubleshooting

### Common Issues

#### Issue: `ImportError: No module named 'strainphase'`

**Cause**: Strainphase not installed or wrong Python environment

**Solution:**
```bash
# Check which Python
which python
python -c "import sys; print(sys.executable)"

# Ensure virtual environment is activated
source strainphase-env/bin/activate

# Reinstall
pip install strainphase[full]
```

#### Issue: `ModuleNotFoundError: No module named 'pysam'`

**Cause**: Optional dependencies not installed

**Solution:**
```bash
# Install full version
pip install strainphase[full]

# Or install pysam separately
pip install pysam
```

#### Issue: `error: Microsoft Visual C++ 14.0 is required` (Windows)

**Cause**: Missing C++ compiler on Windows

**Solution:**
1. Install Build Tools for Visual Studio from https://visualstudio.microsoft.com/downloads/
2. OR use WSL2 (recommended)
3. OR use pre-built wheels: `pip install --only-binary :all: strainphase[full]`

#### Issue: `error: command 'gcc' failed with exit status 1`

**Cause**: Missing build tools on Linux

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum install gcc python3-devel
```

#### Issue: `python-louvain` installation fails

**Solution:**
```bash
# Install alternative package
pip install python-louvain

# OR skip optional dependency (will use connected components instead)
pip install strainphase  # without [full]
```

#### Issue: Tests fail with `FileNotFoundError`

**Cause**: Running tests from wrong directory

**Solution:**
```bash
# Run from package root
cd /path/to/strainphase
strainphase test

# Or use pytest directly
pytest src/strainphase/tests/
```

#### Issue: `OSError: [Errno 24] Too many open files`

**Cause**: System file descriptor limit too low

**Solution:**
```bash
# Check current limit
ulimit -n

# Increase limit (temporary)
ulimit -n 4096

# Permanent fix: Add to ~/.bashrc or /etc/security/limits.conf
echo "* soft nofile 4096" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 8192" | sudo tee -a /etc/security/limits.conf
```

#### Issue: Out of memory during analysis

**Solutions:**
```bash
# Reduce max reads per window
strainphase run --max-reads 100 ...

# Process smaller contigs first
strainphase run --contig small_contig ...

# Process one MAG at a time
strainphase longitudinal --mags MAG_01 ...

# Increase system swap (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Platform-Specific Issues

#### macOS: Command line tools not found

```bash
# Install Xcode command line tools
xcode-select --install
```

#### macOS: SSL certificate errors

```bash
# Install certificates
/Applications/Python\ 3.11/Install\ Certificates.command
```

#### Linux: Missing system libraries

```bash
# Install common libraries
sudo apt-get install -y libz-dev libbz2-dev liblzma-dev
```

### Getting Help

If you encounter issues not covered here:

Email `roles@ucsd.edu` with:
- Operating system and version
- Python version (`python --version`)
- Strainphase version (`strainphase version`)
- Full error message
- Steps to reproduce

## Updating

### Update from PyPI

```bash
# Update to latest version
pip install --upgrade strainphase

# Update with all dependencies
pip install --upgrade strainphase[all]
```

### Update from source

```bash
cd /path/to/strainphase
git pull origin master
pip install -e ".[all]"
```

### Check for updates

```bash
# Current version
strainphase version

# Latest version on PyPI
pip index versions strainphase

# Show outdated packages
pip list --outdated
```

## Uninstalling

### Remove Strainphase

```bash
# Uninstall package
pip uninstall strainphase

# Remove virtual environment (if using venv)
rm -rf strainphase-env/

# Remove conda environment (if using conda)
conda env remove -n strainphase
```

### Clean up

```bash
# Remove cache
rm -rf ~/.cache/pip

# Remove pytest cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type d -name .pytest_cache -exec rm -r {} +
```

## Next Steps

After installation:
1. Read the [Tutorial](tutorials/TUTORIAL.md) for usage examples
2. Check out [Quick Start](../README.md#quick-start) in README
3. Run the test suite to verify everything works
4. (Optional) If you are tuning parameters, run the developer sweep tool (`strainphase sweep --quick`)
