#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Simulate metagenomic reads from strain FASTA files and run variant calling.
#
# Input structure:
#   main_folder/
#     subfolder_001/
#       reference.fasta       <- reference genome
#       strainA.fasta         <- strain genomes (2+)
#       strainB.fasta
#       ...
#     subfolder_002/
#       reference.fasta
#       strainA.fasta
#       strainB.fasta
#       ...
#
# For each subfolder, this pipeline:
#   1. Simulates reads from each strain with badread (pacbio2021 error model)
#   2. Mixes reads at specified abundances across timepoints
#   3. Runs variant calling: minimap2 → Clair3/Longshot → bcftools filter
#   4. Generates ground truth files for validation
#   5. Optionally runs strainphase + validation
#
# Usage:
#   bash benchmarks/simulate_from_fasta.sh \
#       --input-dir /path/to/main_folder \
#       --output-dir /path/to/results \
#       --coverage 30 \
#       --timepoints 4 \
#       --threads 8
#
# Requirements:
#   badread, minimap2, samtools, bcftools, longshot,
#   nanoplot (optional), nanofilt (optional),
#   clair3 (optional, uses longshot if unavailable),
#   nucmer + show-snps (from MUMmer, for ground truth)
# =============================================================================

# ── Defaults ─────────────────────────────────────────────────────────────────

INPUT_DIR=""
OUTPUT_DIR=""
COVERAGE=30
N_TIMEPOINTS=4
THREADS=8
SEED=42
ABUNDANCE_PROFILE="sweep"      # "sweep" or "equal"
ERROR_MODEL="pacbio2021"
QSCORE_MODEL="pacbio2021"
MEAN_READ_LENGTH=15000
READ_LENGTH_STD=3000
VARIANT_CALLER="longshot"      # "longshot" or "clair3"
CLAIR3_MODEL=""                # Path to Clair3 model (required if using clair3)
MIN_QUAL=20
MIN_DEPTH=5
MIN_AF=0.01
SKIP_QC=false
SKIP_STRAINPHASE=false
SKIP_FLORIA=false
SKIP_STRAINY=false
SKIP_VALIDATION=false
DRY_RUN=false
SUBFOLDER=""                   # Process only this subfolder (empty = all)
FLORIA_BIN="floria"
STRAINY_BIN="strainy.py"

# Conda/mamba environment names (empty = use current environment)
ENV_ALIGN=""                   # For badread, minimap2, samtools, bcftools, longshot, etc.
ENV_FLORIA=""                  # For floria
ENV_STRAINY=""                 # For strainy
ENV_STRAINPHASE=""             # For strainphase + validation

STRAINPHASE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Usage ────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Required:
  --input-dir DIR          Main folder containing subfolders with .fasta files
  --output-dir DIR         Output directory for results

Read Simulation:
  --coverage INT           Target total coverage per timepoint (default: $COVERAGE)
  --timepoints INT         Number of timepoints (default: $N_TIMEPOINTS)
  --error-model STR        Badread error model (default: $ERROR_MODEL)
  --qscore-model STR       Badread qscore model (default: $QSCORE_MODEL)
  --mean-read-length INT   Mean read length (default: $MEAN_READ_LENGTH)
  --read-length-std INT    Read length std dev (default: $READ_LENGTH_STD)
  --abundance STR          Abundance profile: "sweep" or "equal" (default: $ABUNDANCE_PROFILE)
  --seed INT               Random seed (default: $SEED)

Variant Calling:
  --variant-caller STR     "longshot" or "clair3" (default: $VARIANT_CALLER)
  --clair3-model DIR       Path to Clair3 model directory (required for clair3)
  --min-qual INT           Minimum variant quality (default: $MIN_QUAL)
  --min-depth INT          Minimum read depth for variants (default: $MIN_DEPTH)
  --min-af FLOAT           Minimum allele frequency (default: $MIN_AF)

Environments (conda/mamba):
  --env-align NAME         Conda env for alignment/variant calling (badread, minimap2, etc.)
  --env-floria NAME        Conda env for Floria
  --env-strainy NAME       Conda env for Strainy
  --env-strainphase NAME   Conda env for strainphase + validation
                           If omitted, the current environment is used for that phase.

Execution:
  --threads INT            Number of threads (default: $THREADS)
  --subfolder NAME         Only process this subfolder (default: all)
  --skip-qc                Skip NanoPlot/NanoFilt QC steps
  --skip-strainphase       Skip strainphase analysis
  --skip-floria            Skip Floria analysis
  --floria-bin PATH        Path to floria binary (default: $FLORIA_BIN)
  --skip-strainy           Skip Strainy analysis
  --strainy-bin PATH       Path to strainy.py (default: $STRAINY_BIN)
  --skip-validation        Skip validation step
  --dry-run                Print what would be done without executing
  --help                   Show this help
EOF
    exit 1
}

# ── Parse arguments ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)          INPUT_DIR="$2"; shift 2 ;;
        --output-dir)         OUTPUT_DIR="$2"; shift 2 ;;
        --coverage)           COVERAGE="$2"; shift 2 ;;
        --timepoints)         N_TIMEPOINTS="$2"; shift 2 ;;
        --error-model)        ERROR_MODEL="$2"; shift 2 ;;
        --qscore-model)       QSCORE_MODEL="$2"; shift 2 ;;
        --mean-read-length)   MEAN_READ_LENGTH="$2"; shift 2 ;;
        --read-length-std)    READ_LENGTH_STD="$2"; shift 2 ;;
        --abundance)          ABUNDANCE_PROFILE="$2"; shift 2 ;;
        --seed)               SEED="$2"; shift 2 ;;
        --variant-caller)     VARIANT_CALLER="$2"; shift 2 ;;
        --clair3-model)       CLAIR3_MODEL="$2"; shift 2 ;;
        --min-qual)           MIN_QUAL="$2"; shift 2 ;;
        --min-depth)          MIN_DEPTH="$2"; shift 2 ;;
        --min-af)             MIN_AF="$2"; shift 2 ;;
        --threads)            THREADS="$2"; shift 2 ;;
        --subfolder)          SUBFOLDER="$2"; shift 2 ;;
        --env-align)          ENV_ALIGN="$2"; shift 2 ;;
        --env-floria)         ENV_FLORIA="$2"; shift 2 ;;
        --env-strainy)        ENV_STRAINY="$2"; shift 2 ;;
        --env-strainphase)    ENV_STRAINPHASE="$2"; shift 2 ;;
        --skip-qc)            SKIP_QC=true; shift ;;
        --skip-strainphase)   SKIP_STRAINPHASE=true; shift ;;
        --skip-floria)        SKIP_FLORIA=true; shift ;;
        --floria-bin)         FLORIA_BIN="$2"; shift 2 ;;
        --skip-strainy)       SKIP_STRAINY=true; shift ;;
        --strainy-bin)        STRAINY_BIN="$2"; shift 2 ;;
        --skip-validation)    SKIP_VALIDATION=true; shift ;;
        --dry-run)            DRY_RUN=true; shift ;;
        --help)               usage ;;
        *)                    echo "ERROR: Unknown option: $1"; usage ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" ]]; then
    echo "ERROR: --input-dir is required"
    usage
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --output-dir is required"
    usage
fi
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi
if [[ "$VARIANT_CALLER" == "clair3" && -z "$CLAIR3_MODEL" ]]; then
    echo "ERROR: --clair3-model is required when using clair3"
    exit 1
fi

# ── Check dependencies ──────────────────────────────────────────────────────

check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "WARNING: $1 not found in PATH"
        return 1
    fi
    return 0
}

# When conda envs are specified, we skip dependency checks for tools in those
# envs (they won't be in the current PATH). We only check tools that will run
# in the current environment (i.e. when no env flag is set for that phase).
echo "Checking dependencies..."
MISSING=0
if [[ -z "$ENV_ALIGN" ]]; then
    for tool in badread minimap2 samtools bcftools; do
        if ! check_tool "$tool"; then
            MISSING=1
        fi
    done
    if [[ "$VARIANT_CALLER" == "longshot" ]]; then
        check_tool longshot || MISSING=1
    elif [[ "$VARIANT_CALLER" == "clair3" ]]; then
        check_tool run_clair3.sh || MISSING=1
    fi
    if [[ "$SKIP_QC" == false ]]; then
        check_tool NanoPlot || echo "  (NanoPlot optional, will skip QC reports)"
        check_tool NanoFilt || echo "  (NanoFilt optional, will skip read filtering)"
    fi
else
    echo "  Alignment tools: will use env '$ENV_ALIGN' (skipping PATH check)"
fi

if [[ "$SKIP_FLORIA" == false ]]; then
    if [[ -z "$ENV_FLORIA" ]]; then
        check_tool "$FLORIA_BIN" || echo "  (Floria not found, use --skip-floria or --floria-bin)"
    else
        echo "  Floria: will use env '$ENV_FLORIA' (skipping PATH check)"
    fi
fi

if [[ "$SKIP_STRAINY" == false ]]; then
    if [[ -z "$ENV_STRAINY" ]]; then
        check_tool "$STRAINY_BIN" || echo "  (Strainy not found, use --skip-strainy or --strainy-bin)"
    else
        echo "  Strainy: will use env '$ENV_STRAINY' (skipping PATH check)"
    fi
fi

if [[ -z "$ENV_STRAINPHASE" ]]; then
    check_tool python3 || MISSING=1
else
    echo "  Strainphase: will use env '$ENV_STRAINPHASE' (skipping PATH check)"
fi

if [[ $MISSING -eq 1 ]]; then
    echo "ERROR: Missing required dependencies. Install them and try again."
    exit 1
fi

# ── Conda/mamba environment support ──────────────────────────────────────────

# Initialize conda shell hooks so `conda activate` works in this script.
# Tries mamba first, then conda.
_conda_initialized=false
_init_conda() {
    if [[ "$_conda_initialized" == true ]]; then
        return 0
    fi
    # Try mamba shell hook first
    if command -v mamba &>/dev/null; then
        eval "$(mamba shell hook -s bash 2>/dev/null)" && _conda_initialized=true && return 0
    fi
    # Fall back to conda
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)" && _conda_initialized=true && return 0
    fi
    # Try sourcing conda.sh from common locations
    for conda_sh in \
        "$CONDA_EXE/../etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [[ -f "$conda_sh" ]]; then
            source "$conda_sh" && _conda_initialized=true && return 0
        fi
    done
    echo "ERROR: Could not initialize conda/mamba. Ensure conda or mamba is installed."
    return 1
}

# Run a command in a specific conda/mamba environment.
# Usage: run_in_env "env_name" command [args...]
# If env_name is empty, runs in the current environment.
run_in_env() {
    local env_name="$1"
    shift
    if [[ -z "$env_name" ]]; then
        "$@"
        return $?
    fi
    (
        _init_conda
        conda activate "$env_name"
        "$@"
    )
    return $?
}

# Run a shell command string in a specific conda/mamba environment.
# Useful for piped commands: run_in_env_sh "myenv" "cmd1 | cmd2"
# If env_name is empty, runs in the current environment.
run_in_env_sh() {
    local env_name="$1"
    local cmd="$2"
    if [[ -z "$env_name" ]]; then
        bash -c "$cmd"
        return $?
    fi
    (
        _init_conda
        conda activate "$env_name"
        bash -c "$cmd"
    )
    return $?
}

# Check if any env flags were set; if so, verify conda/mamba is available
if [[ -n "$ENV_ALIGN" || -n "$ENV_FLORIA" || -n "$ENV_STRAINY" || -n "$ENV_STRAINPHASE" ]]; then
    _init_conda || exit 1
    echo "Conda/mamba environments:"
    [[ -n "$ENV_ALIGN" ]]       && echo "  Alignment:    $ENV_ALIGN"
    [[ -n "$ENV_FLORIA" ]]      && echo "  Floria:       $ENV_FLORIA"
    [[ -n "$ENV_STRAINY" ]]     && echo "  Strainy:      $ENV_STRAINY"
    [[ -n "$ENV_STRAINPHASE" ]] && echo "  Strainphase:  $ENV_STRAINPHASE"
    echo ""
fi

# ── Helper functions ─────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Generate abundance profiles for N strains across T timepoints.
# Outputs lines: strain_index timepoint abundance
# strain_index is 0-based, timepoint is 1-based.
generate_abundances() {
    local n_strains=$1
    local n_timepoints=$2
    local profile=$3
    local seed=$4

    python3 -c "
import numpy as np
rng = np.random.default_rng($seed)
n_strains = $n_strains
n_tp = $n_timepoints
profile = '$profile'

if profile == 'sweep' and n_strains >= 3:
    # strain 0: high -> low, strain 1: low -> high, rest: stable
    high, low = 0.8, 0.05
    remaining = 1.0 - (high + low)
    stable_each = remaining / (n_strains - 2) if n_strains > 2 else 0
    for tp in range(n_tp):
        frac = tp / max(n_tp - 1, 1)
        abunds = [0.0] * n_strains
        abunds[0] = high + (low - high) * frac
        abunds[1] = low + (high - low) * frac
        for i in range(2, n_strains):
            abunds[i] = stable_each
        # Normalize
        total = sum(abunds)
        for i in range(n_strains):
            print(f'{i}\t{tp+1}\t{abunds[i]/total:.6f}')
elif profile == 'equal':
    eq = 1.0 / n_strains
    for tp in range(n_tp):
        for i in range(n_strains):
            print(f'{i}\t{tp+1}\t{eq:.6f}')
else:
    # Random Dirichlet with temporal drift
    base = rng.dirichlet(np.ones(n_strains) * 2)
    for tp in range(n_tp):
        noise = rng.normal(0, 0.05, n_strains) * (tp / max(n_tp, 1))
        abunds = np.clip(base + noise, 0.01, None)
        abunds /= abunds.sum()
        for i in range(n_strains):
            print(f'{i}\t{tp+1}\t{abunds[i]:.6f}')
"
}

# Generate ground truth SNVs by aligning each strain to reference.
# Uses nucmer if available, otherwise minimap2 + paftools.
generate_ground_truth() {
    local ref_fasta=$1
    local output_dir=$2
    shift 2
    local strain_fastas=("$@")

    local n_strains=${#strain_fastas[@]}
    local truth_dir="$output_dir"
    mkdir -p "$truth_dir"

    log "  Generating ground truth SNVs for $n_strains strains"

    # We'll use a Python script for ground truth generation since it needs
    # to produce multiple output files in the strainphase validation format.
    run_in_env "$ENV_STRAINPHASE" \
        python3 "$STRAINPHASE_ROOT/benchmarks/generate_ground_truth.py" \
        --reference "$ref_fasta" \
        --strains "${strain_fastas[@]}" \
        --output-dir "$truth_dir" \
        --abundances "$output_dir/abundances.tsv"
}

# ── Main pipeline per subfolder ──────────────────────────────────────────────

process_subfolder() {
    local subfolder_path=$1
    local subfolder_name
    subfolder_name=$(basename "$subfolder_path")
    local sub_output="$OUTPUT_DIR/$subfolder_name"

    log "============================================================"
    log "Processing: $subfolder_name"
    log "============================================================"

    # ── Identify reference and strain FASTAs ──
    local ref_fasta=""
    local strain_fastas=()
    local strain_names=()

    # Look for reference.fasta (case-insensitive)
    for f in "$subfolder_path"/*.fasta "$subfolder_path"/*.fa "$subfolder_path"/*.fna; do
        [[ -f "$f" ]] || continue
        local fname
        fname=$(basename "$f")
        local fname_lower
        fname_lower=$(echo "$fname" | tr '[:upper:]' '[:lower:]')

        if [[ "$fname_lower" == "reference.fasta" || "$fname_lower" == "reference.fa" || "$fname_lower" == "reference.fna" ]]; then
            ref_fasta="$f"
        else
            strain_fastas+=("$f")
            # Strip extension for strain name
            local sname="${fname%.*}"
            strain_names+=("$sname")
        fi
    done

    if [[ -z "$ref_fasta" ]]; then
        log "WARNING: No reference.fasta found in $subfolder_path — skipping"
        return 1
    fi

    if [[ ${#strain_fastas[@]} -lt 1 ]]; then
        log "WARNING: No strain FASTAs found in $subfolder_path — skipping"
        return 1
    fi

    local n_strains=${#strain_fastas[@]}
    log "  Reference: $(basename "$ref_fasta")"
    log "  Strains ($n_strains): ${strain_names[*]}"

    if [[ "$DRY_RUN" == true ]]; then
        log "  [DRY RUN] Would process $n_strains strains across $N_TIMEPOINTS timepoints"
        return 0
    fi

    mkdir -p "$sub_output"

    # ── Step 1: Generate abundance profiles ──
    log "  Step 1: Generating abundance profiles"

    local abund_file="$sub_output/abundances.tsv"
    {
        echo -e "strain_idx\ttimepoint\tabundance"
        generate_abundances "$n_strains" "$N_TIMEPOINTS" "$ABUNDANCE_PROFILE" "$SEED"
    } > "$abund_file"

    # Also write in strainphase truth format
    local truth_abund_file="$sub_output/truth_abundances.tsv"
    {
        printf "strain_id"
        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            printf "\tT%d" "$tp_idx"
        done
        printf "\n"
        for si in $(seq 0 $((n_strains - 1))); do
            printf "%s" "${strain_names[$si]}"
            for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
                local ab
                ab=$(awk -v s="$si" -v t="$tp_idx" '$1==s && $2==t {print $3}' "$abund_file")
                printf "\t%s" "$ab"
            done
            printf "\n"
        done
    } > "$truth_abund_file"

    log "  Abundance profiles written to $abund_file"

    # ── Step 2: Simulate reads with badread ──
    log "  Step 2: Simulating reads with badread"

    local reads_dir="$sub_output/simulated_reads"
    mkdir -p "$reads_dir"

    for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
        local tp_name="T${tp_idx}"
        local tp_fastq="$reads_dir/${tp_name}.fastq"
        local tp_read_origins="$reads_dir/${tp_name}_read_origins.tsv"

        if [[ -f "$tp_fastq" && -s "$tp_fastq" ]]; then
            log "    $tp_name: Reads already exist — skipping simulation"
            continue
        fi

        log "    $tp_name: Simulating reads..."

        # Clear/create the combined fastq for this timepoint
        > "$tp_fastq"
        > "$tp_read_origins"

        for si in $(seq 0 $((n_strains - 1))); do
            local strain_fasta="${strain_fastas[$si]}"
            local strain_name="${strain_names[$si]}"

            # Get abundance for this strain at this timepoint
            local abundance
            abundance=$(awk -v s="$si" -v t="$tp_idx" '$1==s && $2==t {print $3}' "$abund_file")

            if [[ -z "$abundance" || "$abundance" == "0" || "$abundance" == "0.000000" ]]; then
                continue
            fi

            # Calculate strain-specific coverage
            local strain_cov
            strain_cov=$(python3 -c "print(f'{$COVERAGE * $abundance:.1f}')")

            if (( $(echo "$strain_cov < 0.5" | bc -l) )); then
                log "      $strain_name: coverage ${strain_cov}x too low — skipping"
                continue
            fi

            local strain_fastq="$reads_dir/${tp_name}_${strain_name}.fastq"
            local strain_seed=$((SEED + si * 100 + tp_idx))

            log "      $strain_name: abundance=${abundance}, coverage=${strain_cov}x"

            run_in_env "$ENV_ALIGN" \
                badread simulate \
                --reference "$strain_fasta" \
                --quantity "${strain_cov}x" \
                --error_model "$ERROR_MODEL" \
                --qscore_model "$QSCORE_MODEL" \
                --length "$MEAN_READ_LENGTH,$READ_LENGTH_STD" \
                --seed "$strain_seed" \
                > "$strain_fastq" \
                2>> "$reads_dir/${tp_name}_badread.log"

            # Append to combined fastq
            cat "$strain_fastq" >> "$tp_fastq"

            # Record read origins (read_id -> strain)
            grep "^@" "$strain_fastq" | sed 's/^@//' | while read -r read_header; do
                local read_id
                read_id=$(echo "$read_header" | awk '{print $1}')
                echo -e "${read_id}\t${strain_name}" >> "$tp_read_origins"
            done

            # Clean up per-strain fastq
            rm -f "$strain_fastq"
        done

        local n_reads
        n_reads=$(grep -c "^@" "$tp_fastq" || true)
        log "    $tp_name: Generated $n_reads reads"
    done

    # ── Step 3: QC (optional) ──
    if [[ "$SKIP_QC" == false ]]; then
        log "  Step 3: Running QC"
        local qc_dir="$sub_output/qc"
        mkdir -p "$qc_dir"

        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            local tp_name="T${tp_idx}"
            local tp_fastq="$reads_dir/${tp_name}.fastq"
            local filtered_fastq="$reads_dir/${tp_name}_filtered.fastq"

            # NanoPlot (if available)
            if run_in_env "$ENV_ALIGN" command -v NanoPlot &>/dev/null; then
                if [[ ! -d "$qc_dir/${tp_name}" ]]; then
                    log "    $tp_name: Running NanoPlot..."
                    run_in_env "$ENV_ALIGN" \
                        NanoPlot --fastq "$tp_fastq" \
                        -o "$qc_dir/${tp_name}" \
                        -t "$THREADS" \
                        --no_static \
                        2>> "$qc_dir/${tp_name}_nanoplot.log" || true
                fi
            fi

            # NanoFilt (if available)
            if run_in_env "$ENV_ALIGN" command -v NanoFilt &>/dev/null; then
                if [[ ! -f "$filtered_fastq" ]]; then
                    log "    $tp_name: Running NanoFilt (q>=10, len>=1000)..."
                    run_in_env_sh "$ENV_ALIGN" \
                        "NanoFilt -q 10 -l 1000 < '$tp_fastq' > '$filtered_fastq' 2>> '$qc_dir/${tp_name}_nanofilt.log'" \
                        || cp "$tp_fastq" "$filtered_fastq"
                fi
            else
                # No filtering, just copy
                if [[ ! -f "$filtered_fastq" ]]; then
                    cp "$tp_fastq" "$filtered_fastq"
                fi
            fi
        done
    else
        log "  Step 3: Skipping QC"
        # Create symlinks for filtered fastq
        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            local tp_name="T${tp_idx}"
            local tp_fastq="$reads_dir/${tp_name}.fastq"
            local filtered_fastq="$reads_dir/${tp_name}_filtered.fastq"
            if [[ ! -f "$filtered_fastq" ]]; then
                cp "$tp_fastq" "$filtered_fastq"
            fi
        done
    fi

    # ── Step 4: Align reads to reference ──
    log "  Step 4: Aligning reads with minimap2"

    # Index reference
    if [[ ! -f "${ref_fasta}.fai" ]]; then
        run_in_env "$ENV_ALIGN" samtools faidx "$ref_fasta"
    fi

    local bam_dir="$sub_output/alignments"
    mkdir -p "$bam_dir"

    for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
        local tp_name="T${tp_idx}"
        local input_fastq="$reads_dir/${tp_name}_filtered.fastq"
        local bam_file="$bam_dir/${tp_name}.bam"

        if [[ -f "$bam_file" && -f "${bam_file}.bai" ]]; then
            log "    $tp_name: BAM already exists — skipping alignment"
            continue
        fi

        log "    $tp_name: Aligning..."

        run_in_env_sh "$ENV_ALIGN" \
            "minimap2 -a -x map-hifi -t $THREADS --MD -R '@RG\tID:${tp_name}\tSM:${tp_name}\tPL:PACBIO' '$ref_fasta' '$input_fastq' 2>> '$bam_dir/${tp_name}_minimap2.log' | samtools sort -@ $THREADS -o '$bam_file' -"

        run_in_env "$ENV_ALIGN" samtools index "$bam_file"

        local n_mapped
        n_mapped=$(run_in_env "$ENV_ALIGN" samtools view -c -F 4 "$bam_file")
        log "    $tp_name: $n_mapped mapped reads"
    done

    # Copy reference to output for strainphase
    cp "$ref_fasta" "$sub_output/reference.fasta"
    run_in_env "$ENV_ALIGN" samtools faidx "$sub_output/reference.fasta"

    # ── Step 5: Variant calling ──
    log "  Step 5: Calling variants with $VARIANT_CALLER"

    local vcf_dir="$sub_output/variants"
    mkdir -p "$vcf_dir"

    # Merge all timepoint BAMs for variant calling
    local merged_bam="$vcf_dir/all_timepoints.bam"
    if [[ ! -f "$merged_bam" ]]; then
        local bam_list=()
        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            bam_list+=("$bam_dir/T${tp_idx}.bam")
        done

        if [[ ${#bam_list[@]} -eq 1 ]]; then
            cp "${bam_list[0]}" "$merged_bam"
        else
            run_in_env "$ENV_ALIGN" samtools merge -f "$merged_bam" "${bam_list[@]}"
        fi
        run_in_env "$ENV_ALIGN" samtools index "$merged_bam"
    fi

    local raw_vcf="$vcf_dir/raw_variants.vcf"
    local filtered_vcf="$vcf_dir/filtered_variants.vcf"
    local final_vcf="$sub_output/variants.vcf"

    if [[ "$VARIANT_CALLER" == "clair3" ]]; then
        # ── Clair3 ──
        local clair3_dir="$vcf_dir/clair3_output"
        if [[ ! -d "$clair3_dir" ]]; then
            log "    Running Clair3..."
            run_in_env "$ENV_ALIGN" \
                run_clair3.sh \
                --bam_fn="$merged_bam" \
                --ref_fn="$sub_output/reference.fasta" \
                --output="$clair3_dir" \
                --threads="$THREADS" \
                --platform="hifi" \
                --model_path="$CLAIR3_MODEL" \
                --sample_name="sample" \
                --include_all_ctgs \
                2>> "$vcf_dir/clair3.log"
        fi

        # Extract SNVs from Clair3 output
        if [[ -f "$clair3_dir/merge_output.vcf.gz" ]]; then
            run_in_env "$ENV_ALIGN" bcftools view -v snps "$clair3_dir/merge_output.vcf.gz" > "$raw_vcf"
        elif [[ -f "$clair3_dir/pileup.vcf.gz" ]]; then
            run_in_env "$ENV_ALIGN" bcftools view -v snps "$clair3_dir/pileup.vcf.gz" > "$raw_vcf"
        else
            log "    WARNING: No Clair3 output VCF found"
            return 1
        fi

    elif [[ "$VARIANT_CALLER" == "longshot" ]]; then
        # ── Longshot ──
        if [[ ! -f "$raw_vcf" ]]; then
            log "    Running Longshot..."
            run_in_env "$ENV_ALIGN" \
                longshot \
                --bam "$merged_bam" \
                --ref "$sub_output/reference.fasta" \
                --out "$raw_vcf" \
                --min_cov "$MIN_DEPTH" \
                --min_alt_count 2 \
                --min_allele_qual "$MIN_QUAL" \
                2>> "$vcf_dir/longshot.log" || {
                    log "    WARNING: Longshot failed, falling back to pileup caller"
                    # Fallback: use bcftools mpileup
                    run_in_env_sh "$ENV_ALIGN" \
                        "bcftools mpileup -f '$sub_output/reference.fasta' -q $MIN_QUAL -Q 20 --max-depth 10000 '$merged_bam' | bcftools call -mv --ploidy 1 -Ov > '$raw_vcf' 2>> '$vcf_dir/mpileup.log'"
                }
        fi
    fi

    # ── Filter variants ──
    if [[ -f "$raw_vcf" && ! -f "$filtered_vcf" ]]; then
        log "    Filtering variants (QUAL>=${MIN_QUAL}, DP>=${MIN_DEPTH})..."

        run_in_env_sh "$ENV_ALIGN" \
            "bcftools view -i 'QUAL>=${MIN_QUAL}' '$raw_vcf' | bcftools view -v snps | bcftools norm -f '$sub_output/reference.fasta' -d snps > '$filtered_vcf' 2>> '$vcf_dir/filter.log'"

        local n_variants
        n_variants=$(grep -cv "^#" "$filtered_vcf" || true)
        log "    $n_variants SNVs after filtering"
    fi

    # Copy final VCF to output root
    if [[ -f "$filtered_vcf" ]]; then
        cp "$filtered_vcf" "$final_vcf"
        run_in_env_sh "$ENV_ALIGN" "bgzip -c '$final_vcf' > '${final_vcf}.gz'"
        run_in_env "$ENV_ALIGN" tabix -p vcf "${final_vcf}.gz"
    fi

    # Symlink per-timepoint BAMs to output root (strainphase expects T1.bam etc.)
    for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
        local tp_name="T${tp_idx}"
        local src_bam="$bam_dir/${tp_name}.bam"
        local dst_bam="$sub_output/${tp_name}.bam"
        if [[ -f "$src_bam" && ! -f "$dst_bam" ]]; then
            ln -sf "$(realpath "$src_bam")" "$dst_bam"
            ln -sf "$(realpath "${src_bam}.bai")" "${dst_bam}.bai"
        fi
    done

    # ── Step 6: Generate ground truth ──
    log "  Step 6: Generating ground truth"

    generate_ground_truth "$ref_fasta" "$sub_output" "${strain_fastas[@]}"

    # ── Step 7: Run strainphase (optional) ──
    if [[ "$SKIP_STRAINPHASE" == false ]]; then
        log "  Step 7: Running strainphase"

        local sp_output="$sub_output/strainphase_output"
        mkdir -p "$sp_output"

        local bam_args=()
        local vcf_args=()
        local tp_args=()
        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            bam_args+=("$sub_output/T${tp_idx}.bam")
            vcf_args+=("$sub_output/variants.vcf.gz")
            tp_args+=("T${tp_idx}")
        done

        run_in_env "$ENV_STRAINPHASE" \
            python3 -m strainphase.core \
            --bam-paths "${bam_args[@]}" \
            --vcf-paths "${vcf_args[@]}" \
            --reference "$sub_output/reference.fasta" \
            --timepoints "${tp_args[@]}" \
            --output "$sp_output" \
            2>&1 | tee "$sp_output/strainphase.log" || {
                log "    WARNING: strainphase failed for $subfolder_name"
            }
    else
        log "  Step 7: Skipping strainphase"
    fi

    # ── Step 8: Run Floria (optional) ──
    if [[ "$SKIP_FLORIA" == false ]]; then
        log "  Step 8: Running Floria"

        local floria_dir="$sub_output/floria_output"
        mkdir -p "$floria_dir"

        # 8a: Run Floria per timepoint
        log "    Running Floria per timepoint..."
        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            local tp_name="T${tp_idx}"
            local tp_bam="$sub_output/${tp_name}.bam"
            local floria_tp_out="$floria_dir/floria_${tp_name}"

            if [[ ! -f "$tp_bam" ]]; then
                log "      WARNING: BAM not found for $tp_name: $tp_bam — skipping"
                continue
            fi

            if [[ -d "$floria_tp_out" && -f "$floria_tp_out/cmd.log" ]]; then
                log "      $tp_name: Floria output already exists — skipping"
            else
                log "      $tp_name: Running floria..."
                mkdir -p "$floria_tp_out"
                run_in_env "$ENV_FLORIA" \
                    "$FLORIA_BIN" \
                    -b "$tp_bam" \
                    -v "$sub_output/variants.vcf.gz" \
                    -r "$sub_output/reference.fasta" \
                    -o "$floria_tp_out" \
                    --overwrite \
                    2>&1 | tee "$floria_tp_out/floria_stdout.log" || {
                        log "      WARNING: Floria failed for $tp_name"
                    }
                log "      $tp_name: Floria complete"
            fi
        done

        # 8b: Convert each Floria output to per-timepoint lineages.tsv
        log "    Converting Floria output to lineages.tsv..."
        local convert_dir="$floria_dir/converted"
        mkdir -p "$convert_dir"

        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            local tp_name="T${tp_idx}"
            local floria_tp_out="$floria_dir/floria_${tp_name}"
            local tp_convert_dir="$convert_dir/${tp_name}"

            if [[ ! -d "$floria_tp_out" ]]; then
                log "      $tp_name: No Floria output — skipping"
                continue
            fi

            log "      $tp_name: Converting..."
            run_in_env "$ENV_STRAINPHASE" \
                python3 "$STRAINPHASE_ROOT/validation/convert_floria.py" \
                --floria-dir "$floria_tp_out" \
                --vcf "$sub_output/variants.vcf.gz" \
                --sample "$tp_name" \
                --output-dir "$tp_convert_dir" \
                2>> "$floria_dir/convert.log" || {
                    log "      WARNING: Conversion failed for $tp_name"
                    continue
                }
            log "      $tp_name: Wrote $tp_convert_dir/lineages.tsv"
        done

        # 8c: Combine per-timepoint lineages into one file
        log "    Combining lineages across timepoints..."
        local floria_combined="$floria_dir/lineages.tsv"
        local header_written=false

        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            local tp_name="T${tp_idx}"
            local tp_file="$convert_dir/${tp_name}/lineages.tsv"

            if [[ ! -f "$tp_file" ]]; then
                log "      $tp_name: No lineages.tsv — skipping"
                continue
            fi

            if [[ "$header_written" == false ]]; then
                head -1 "$tp_file" > "$floria_combined"
                header_written=true
            fi

            # Append data rows, prefixing lineage_id and track_id with timepoint
            tail -n +2 "$tp_file" | while IFS=$'\t' read -r lineage_id sample contig track_id rest; do
                printf '%s\t%s\t%s\t%s\t%s\n' \
                    "${tp_name}_${lineage_id}" "$sample" "$contig" "${tp_name}_${track_id}" "$rest"
            done >> "$floria_combined"
        done

        if [[ -f "$floria_combined" ]]; then
            local n_floria_lines=$(( $(wc -l < "$floria_combined") - 1 ))
            log "    Combined $n_floria_lines Floria haplotypes into $floria_combined"
        fi
    else
        log "  Step 8: Skipping Floria"
    fi

    # ── Step 9: Run Strainy (optional) ──
    if [[ "$SKIP_STRAINY" == false ]]; then
        log "  Step 9: Running Strainy"

        local strainy_dir="$sub_output/strainy_output"
        mkdir -p "$strainy_dir"

        # 9a: Convert FASTA reference to GFA (strainy requires GFA input)
        local ref_gfa="$sub_output/reference.gfa"
        if [[ ! -f "$ref_gfa" ]]; then
            log "    Converting reference FASTA to GFA..."
            python3 - "$sub_output/reference.fasta" "$ref_gfa" <<'PYEOF'
import sys
fasta_path, gfa_path = sys.argv[1], sys.argv[2]
with open(fasta_path) as fin, open(gfa_path, "w") as fout:
    name, seq = None, []
    fout.write("H\tVN:Z:1.0\n")
    for line in fin:
        line = line.strip()
        if line.startswith(">"):
            if name:
                fout.write(f"S\t{name}\t{''.join(seq)}\n")
            name = line[1:].split()[0]
            seq = []
        else:
            seq.append(line)
    if name:
        fout.write(f"S\t{name}\t{''.join(seq)}\n")
PYEOF
        fi

        # 9b: Combine all timepoint reads into a single fastq for strainy
        # (strainy runs on all reads at once, using the BAM for phasing)
        local combined_fastq="$sub_output/all_reads.fastq"
        if [[ ! -f "$combined_fastq" ]]; then
            log "    Combining reads across timepoints..."
            > "$combined_fastq"
            for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
                local tp_fastq="$reads_dir/T${tp_idx}_filtered.fastq"
                if [[ -f "$tp_fastq" ]]; then
                    cat "$tp_fastq" >> "$combined_fastq"
                fi
            done
        fi

        # 9c: Run Strainy per timepoint
        log "    Running Strainy per timepoint..."
        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            local tp_name="T${tp_idx}"
            local tp_bam="$sub_output/${tp_name}.bam"
            local tp_fastq="$reads_dir/${tp_name}_filtered.fastq"
            local strainy_tp_out="$strainy_dir/strainy_${tp_name}"

            if [[ ! -f "$tp_bam" ]]; then
                log "      WARNING: BAM not found for $tp_name: $tp_bam — skipping"
                continue
            fi

            if [[ -d "$strainy_tp_out" && -f "$strainy_tp_out/strainy_final.gfa" ]]; then
                log "      $tp_name: Strainy output already exists — skipping"
            else
                log "      $tp_name: Running strainy..."
                mkdir -p "$strainy_tp_out"
                run_in_env "$ENV_STRAINY" \
                    "$STRAINY_BIN" \
                    --gfa_ref "$ref_gfa" \
                    --fastq "$tp_fastq" \
                    --mode hifi \
                    --bam "$tp_bam" \
                    --snp "$sub_output/variants.vcf" \
                    --unitig-split-length 0 \
                    --stage phase \
                    --output "$strainy_tp_out" \
                    2>&1 | tee "$strainy_tp_out/strainy_stdout.log" || {
                        log "      WARNING: Strainy failed for $tp_name"
                    }
                log "      $tp_name: Strainy complete"
            fi
        done

        # 9d: Convert each Strainy output to per-timepoint lineages.tsv
        log "    Converting Strainy output to lineages.tsv..."
        local strainy_convert_dir="$strainy_dir/converted"
        mkdir -p "$strainy_convert_dir"

        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            local tp_name="T${tp_idx}"
            local strainy_tp_out="$strainy_dir/strainy_${tp_name}"
            local tp_convert_dir="$strainy_convert_dir/${tp_name}"

            if [[ ! -d "$strainy_tp_out" ]]; then
                log "      $tp_name: No Strainy output — skipping"
                continue
            fi

            log "      $tp_name: Converting..."
            run_in_env "$ENV_STRAINPHASE" \
                python3 "$STRAINPHASE_ROOT/validation/convert_strainy.py" \
                --strainy-dir "$strainy_tp_out" \
                --vcf "$sub_output/variants.vcf" \
                --sample "$tp_name" \
                --output-dir "$tp_convert_dir" \
                2>> "$strainy_dir/convert.log" || {
                    log "      WARNING: Conversion failed for $tp_name"
                    continue
                }
            log "      $tp_name: Wrote $tp_convert_dir/lineages.tsv"
        done

        # 9e: Combine per-timepoint lineages into one file
        log "    Combining lineages across timepoints..."
        local strainy_combined="$strainy_dir/lineages.tsv"
        local strainy_header_written=false

        for tp_idx in $(seq 1 "$N_TIMEPOINTS"); do
            local tp_name="T${tp_idx}"
            local tp_file="$strainy_convert_dir/${tp_name}/lineages.tsv"

            if [[ ! -f "$tp_file" ]]; then
                log "      $tp_name: No lineages.tsv — skipping"
                continue
            fi

            if [[ "$strainy_header_written" == false ]]; then
                head -1 "$tp_file" > "$strainy_combined"
                strainy_header_written=true
            fi

            # Append data rows, prefixing lineage_id and track_id with timepoint
            tail -n +2 "$tp_file" | while IFS=$'\t' read -r lineage_id sample contig track_id rest; do
                printf '%s\t%s\t%s\t%s\t%s\n' \
                    "${tp_name}_${lineage_id}" "$sample" "$contig" "${tp_name}_${track_id}" "$rest"
            done >> "$strainy_combined"
        done

        if [[ -f "$strainy_combined" ]]; then
            local n_strainy_lines=$(( $(wc -l < "$strainy_combined") - 1 ))
            log "    Combined $n_strainy_lines Strainy haplotypes into $strainy_combined"
        fi
    else
        log "  Step 9: Skipping Strainy"
    fi

    # ── Step 10: Validate strainphase output (optional) ──
    if [[ "$SKIP_VALIDATION" == false && "$SKIP_STRAINPHASE" == false ]]; then
        log "  Step 10: Validating strainphase output"

        local sp_lineages="$sub_output/strainphase_output/lineages.tsv"
        local sp_val_output="$sub_output/validation_strainphase"
        mkdir -p "$sp_val_output"

        if [[ -f "$sp_lineages" ]]; then
            run_in_env "$ENV_STRAINPHASE" \
                python3 -m validation.validate_haplotypes \
                --detected "$sp_lineages" \
                --truth "$sub_output" \
                --output "$sp_val_output" \
                2>&1 | tee "$sp_val_output/validation.log" || {
                    log "    WARNING: Strainphase validation failed for $subfolder_name"
                }
        else
            log "    WARNING: No strainphase lineages.tsv found — skipping validation"
        fi
    else
        log "  Step 10: Skipping strainphase validation"
    fi

    # ── Step 11: Validate Floria output (optional) ──
    if [[ "$SKIP_VALIDATION" == false && "$SKIP_FLORIA" == false ]]; then
        log "  Step 11: Validating Floria output"

        local floria_lineages="$sub_output/floria_output/lineages.tsv"
        local floria_val_output="$sub_output/validation_floria"
        mkdir -p "$floria_val_output"

        if [[ -f "$floria_lineages" ]]; then
            run_in_env "$ENV_STRAINPHASE" \
                python3 -m validation.validate_haplotypes \
                --detected "$floria_lineages" \
                --truth "$sub_output" \
                --output "$floria_val_output" \
                2>&1 | tee "$floria_val_output/validation.log" || {
                    log "    WARNING: Floria validation failed for $subfolder_name"
                }
        else
            log "    WARNING: No Floria lineages.tsv found — skipping validation"
        fi
    else
        log "  Step 11: Skipping Floria validation"
    fi

    # ── Step 12: Validate Strainy output (optional) ──
    if [[ "$SKIP_VALIDATION" == false && "$SKIP_STRAINY" == false ]]; then
        log "  Step 12: Validating Strainy output"

        local strainy_lineages="$sub_output/strainy_output/lineages.tsv"
        local strainy_val_output="$sub_output/validation_strainy"
        mkdir -p "$strainy_val_output"

        if [[ -f "$strainy_lineages" ]]; then
            run_in_env "$ENV_STRAINPHASE" \
                python3 -m validation.validate_haplotypes \
                --detected "$strainy_lineages" \
                --truth "$sub_output" \
                --output "$strainy_val_output" \
                2>&1 | tee "$strainy_val_output/validation.log" || {
                    log "    WARNING: Strainy validation failed for $subfolder_name"
                }
        else
            log "    WARNING: No Strainy lineages.tsv found — skipping validation"
        fi
    else
        log "  Step 12: Skipping Strainy validation"
    fi

    log "  Done: $subfolder_name"
    log ""
}

# ── Main ─────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Strain FASTA Simulation Pipeline"
echo "============================================================"
echo "  Input:          $INPUT_DIR"
echo "  Output:         $OUTPUT_DIR"
echo "  Coverage:       ${COVERAGE}x"
echo "  Timepoints:     $N_TIMEPOINTS"
echo "  Error model:    $ERROR_MODEL"
echo "  Abundance:      $ABUNDANCE_PROFILE"
echo "  Variant caller: $VARIANT_CALLER"
echo "  Threads:        $THREADS"
echo "  Skip QC:        $SKIP_QC"
echo "  Skip strainphase: $SKIP_STRAINPHASE"
echo "  Skip Floria:      $SKIP_FLORIA"
echo "  Floria bin:       $FLORIA_BIN"
echo "  Skip Strainy:     $SKIP_STRAINY"
echo "  Strainy bin:      $STRAINY_BIN"
echo "  Skip validation:  $SKIP_VALIDATION"
echo "  Env (align):      ${ENV_ALIGN:-(current)}"
echo "  Env (floria):     ${ENV_FLORIA:-(current)}"
echo "  Env (strainy):    ${ENV_STRAINY:-(current)}"
echo "  Env (strainphase): ${ENV_STRAINPHASE:-(current)}"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# Collect subfolders
SUBFOLDERS=()
if [[ -n "$SUBFOLDER" ]]; then
    if [[ -d "$INPUT_DIR/$SUBFOLDER" ]]; then
        SUBFOLDERS=("$INPUT_DIR/$SUBFOLDER")
    else
        echo "ERROR: Subfolder not found: $INPUT_DIR/$SUBFOLDER"
        exit 1
    fi
else
    for d in "$INPUT_DIR"/*/; do
        [[ -d "$d" ]] && SUBFOLDERS+=("${d%/}")
    done
fi

if [[ ${#SUBFOLDERS[@]} -eq 0 ]]; then
    echo "ERROR: No subfolders found in $INPUT_DIR"
    exit 1
fi

log "Found ${#SUBFOLDERS[@]} subfolders to process"
echo ""

# Track results
RESULTS_FILE="$OUTPUT_DIR/pipeline_results.tsv"
echo -e "subfolder\tn_strains\tn_variants\tstatus" > "$RESULTS_FILE"

TOTAL=${#SUBFOLDERS[@]}
PROCESSED=0
FAILED=0

for subfolder_path in "${SUBFOLDERS[@]}"; do
    PROCESSED=$((PROCESSED + 1))
    subfolder_name=$(basename "$subfolder_path")
    log "[$PROCESSED/$TOTAL] $subfolder_name"

    if process_subfolder "$subfolder_path"; then
        # Count strains and variants
        n_strains=$(ls "$subfolder_path"/*.fasta "$subfolder_path"/*.fa "$subfolder_path"/*.fna 2>/dev/null \
            | grep -icv "reference" || true)
        n_variants=0
        if [[ -f "$OUTPUT_DIR/$subfolder_name/variants.vcf" ]]; then
            n_variants=$(grep -cv "^#" "$OUTPUT_DIR/$subfolder_name/variants.vcf" || true)
        fi
        echo -e "$subfolder_name\t$n_strains\t$n_variants\tOK" >> "$RESULTS_FILE"
    else
        FAILED=$((FAILED + 1))
        echo -e "$subfolder_name\t0\t0\tFAILED" >> "$RESULTS_FILE"
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
log "============================================================"
log "PIPELINE COMPLETE"
log "============================================================"
log "  Processed: $PROCESSED subfolders"
log "  Failed:    $FAILED"
log "  Results:   $RESULTS_FILE"
log "  Output:    $OUTPUT_DIR"
log "============================================================"

# Print per-subfolder summary
if [[ "$DRY_RUN" == false ]]; then
    echo ""
    echo "Per-subfolder results:"
    column -t -s $'\t' "$RESULTS_FILE"
fi

# Aggregate validation metrics if available
if [[ "$SKIP_VALIDATION" == false && "$DRY_RUN" == false ]]; then
    echo ""
    log "Aggregating validation metrics..."
    python3 -c "
import json, os, glob
import numpy as np

results_dir = '$OUTPUT_DIR'
skip_sp = '$SKIP_STRAINPHASE' == 'true'
skip_fl = '$SKIP_FLORIA' == 'true'
skip_st = '$SKIP_STRAINY' == 'true'

all_results = {}

def collect_metrics(pattern):
    metrics = []
    for val_json in sorted(glob.glob(os.path.join(results_dir, pattern))):
        subfolder = os.path.basename(os.path.dirname(os.path.dirname(val_json)))
        with open(val_json) as f:
            m = json.load(f)
        metrics.append({
            'subfolder': subfolder,
            'precision': m.get('precision', 0),
            'recall': m.get('recall', 0),
            'f1': m.get('f1', 0),
            'snv_precision': m.get('snv_precision', 0),
            'snv_recall': m.get('snv_recall', 0),
        })
    return metrics

if not skip_sp:
    all_results['strainphase'] = collect_metrics('*/validation_strainphase/validation_metrics.json')
if not skip_fl:
    all_results['floria'] = collect_metrics('*/validation_floria/validation_metrics.json')
if not skip_st:
    all_results['strainy'] = collect_metrics('*/validation_strainy/validation_metrics.json')

for tool_name, metrics in all_results.items():
    if metrics:
        print(f'\\n{tool_name.upper()} — Aggregated over {len(metrics)} subfolders:')
        for key in ['precision', 'recall', 'f1', 'snv_precision', 'snv_recall']:
            vals = [m[key] for m in metrics]
            print(f'  {key:20s}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}')
    else:
        print(f'\\n{tool_name.upper()} — No validation metrics found.')

# Save aggregated metrics
if any(all_results.values()):
    agg_file = os.path.join(results_dir, 'aggregated_metrics.json')
    with open(agg_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\\nSaved to {agg_file}')

# Side-by-side comparison across all tools that ran
tool_names = [t for t in all_results if all_results[t]]
if len(tool_names) >= 2:
    tool_data = {t: {m['subfolder']: m for m in all_results[t]} for t in tool_names}
    common = sorted(set.intersection(*[set(d.keys()) for d in tool_data.values()]))
    if common:
        print(f'\\nSIDE-BY-SIDE COMPARISON ({len(common)} subfolders):')
        header = f'{\"\":20s}' + ''.join(f'  {t:>12s}' for t in tool_names)
        print(header)
        for key in ['precision', 'recall', 'f1', 'snv_precision', 'snv_recall']:
            row = f'  {key:20s}:'
            for t in tool_names:
                vals = [tool_data[t][s][key] for s in common]
                row += f'  {np.mean(vals):12.3f}'
            print(row)
" 2>/dev/null || true
fi
