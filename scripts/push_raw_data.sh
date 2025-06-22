#!/bin/bash

# push raw data for generate_cluster_dataset script
# usage: ./scripts/push_raw_data.sh "ssh -p 44386 root@24.124.32.70 -L 8080:localhost:8080"

set -e  # exit on any error

# colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # no color

# function to print colored messages
print_msg() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%H:%M:%S')] ${message}${NC}"
}

print_success() { print_msg "$GREEN" "✓ $1"; }
print_info() { print_msg "$BLUE" "ℹ $1"; }
print_warning() { print_msg "$YELLOW" "⚠ $1"; }
print_error() { print_msg "$RED" "✗ $1"; }

# function to show usage
show_usage() {
    echo "Usage: $0 \"SSH_CONNECTION_STRING\""
    echo ""
    echo "Example:"
    echo "  $0 \"ssh -p 44386 root@24.124.32.70 -L 8080:localhost:8080\""
    echo ""
    echo "This script transfers the raw data files needed for generate_cluster_dataset:"
    echo "  - data/processed/trips_with_weather.parquet (378MB)"
    echo "  - data/raw/combined/trips.parquet (499MB)"
    echo "  - data/raw/combined/users.parquet (5.4MB)"
    echo ""
}

# check arguments
if [ $# -ne 1 ]; then
    print_error "Missing SSH connection string"
    show_usage
    exit 1
fi

SSH_CMD="$1"

# extract connection details from ssh command
if [[ $SSH_CMD =~ ssh[[:space:]]+(.*[[:space:]])?([^@]+@[^[:space:]]+) ]]; then
    SSH_HOST="${BASH_REMATCH[2]}"
    SSH_OPTIONS="${BASH_REMATCH[1]}"
    
    # remove port forwarding option for non-interactive commands
    SSH_CMD_CLEAN=$(echo "$SSH_CMD" | sed 's/-L[[:space:]]*[^[:space:]]*//g')
else
    print_error "Could not parse SSH connection string"
    show_usage
    exit 1
fi

print_info "Starting raw data transfer for EcoBici-AI"
print_info "SSH Host: $SSH_HOST"
print_info "SSH Options: $SSH_OPTIONS"

# define required files with their sizes (for verification)
declare -A REQUIRED_FILES=(
    ["data/processed/trips_with_weather.parquet"]="378MB"
    ["data/raw/combined/trips.parquet"]="499MB"
    ["data/raw/combined/users.parquet"]="5.4MB"
)

# verify local files exist
print_info "Verifying local files..."
missing_files=0

for file in "${!REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        # use quoted variable to prevent bash from interpreting special characters
        size=$(du -h "${file}" | cut -f1)
        print_success "Found: ${file} (${size})"
    else
        print_error "Missing: ${file}"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -gt 0 ]; then
    print_error "Missing $missing_files required files. Cannot proceed."
    exit 1
fi

# test ssh connection
print_info "Testing SSH connection..."
if $SSH_CMD_CLEAN "echo 'Connection successful'" > /dev/null 2>&1; then
    print_success "SSH connection established"
else
    print_error "SSH connection failed"
    print_info "Please verify your SSH connection string and network connectivity"
    exit 1
fi

# get remote working directory
print_info "Getting remote working directory..."
REMOTE_PWD=$($SSH_CMD_CLEAN "pwd")
print_info "Remote working directory: $REMOTE_PWD"

# create remote directory structure
print_info "Creating remote directory structure..."
$SSH_CMD_CLEAN "mkdir -p data/processed data/raw/combined"
print_success "Remote directories created"

# function to transfer file with progress and verification
transfer_file() {
    local local_file="$1"
    local remote_path="$2"
    local expected_size="$3"
    
    print_info "Transferring: $local_file -> $remote_path"
    
    # extract scp options from ssh command
    SCP_OPTIONS=""
    if [[ $SSH_OPTIONS =~ -p[[:space:]]+([0-9]+) ]]; then
        SCP_OPTIONS="-P ${BASH_REMATCH[1]}"
    elif [[ $SSH_OPTIONS =~ -p([0-9]+) ]]; then
        SCP_OPTIONS="-P ${BASH_REMATCH[1]}"
    fi
    
    # perform transfer with progress
    if scp $SCP_OPTIONS "$local_file" "$SSH_HOST:$remote_path" 2>&1; then
        print_success "Transfer completed: $local_file"
        
        # verify remote file exists and get size
        remote_size=$($SSH_CMD_CLEAN "du -h '$remote_path' 2>/dev/null | cut -f1" || echo "FAILED")
        
        if [ "$remote_size" = "FAILED" ]; then
            print_error "Failed to verify remote file: $remote_path"
            return 1
        else
            print_success "Remote file verified: $remote_path ($remote_size)"
            return 0
        fi
    else
        print_error "Transfer failed: $local_file"
        return 1
    fi
}

# transfer all required files
print_info "Starting file transfers..."
transfer_errors=0

# transfer processed data
if ! transfer_file "data/processed/trips_with_weather.parquet" "data/processed/trips_with_weather.parquet" "378MB"; then
    transfer_errors=$((transfer_errors + 1))
fi

# transfer raw data
if ! transfer_file "data/raw/combined/trips.parquet" "data/raw/combined/trips.parquet" "499MB"; then
    transfer_errors=$((transfer_errors + 1))
fi

if ! transfer_file "data/raw/combined/users.parquet" "data/raw/combined/users.parquet" "5.4MB"; then
    transfer_errors=$((transfer_errors + 1))
fi

# create .gitkeep file in raw directory
print_info "Creating .gitkeep file in remote raw directory..."
$SSH_CMD_CLEAN "touch data/raw/.gitkeep"
print_success ".gitkeep file created"

# final verification
print_info "Performing final verification..."
$SSH_CMD_CLEAN "
echo 'Remote data structure:'
find data -type f -name '*.parquet' -exec ls -lh {} \; 2>/dev/null || echo 'No parquet files found'
echo ''
echo 'Directory structure:'
find data -type d | sort
echo ''
echo 'Total data size:'
du -sh data 2>/dev/null || echo 'Could not calculate total size'
"

if [ $transfer_errors -eq 0 ]; then
    print_success "All files transferred successfully!"
    print_info "You can now run generate_cluster_dataset on the remote server:"
    print_info "  cd /workspace/EcoBici-AI"
    print_info "  python scripts/generate_cluster_dataset.py --data_dir data --output_dir data/clustered"
else
    print_error "$transfer_errors file transfers failed"
    exit 1
fi

print_success "Raw data transfer completed successfully!" 