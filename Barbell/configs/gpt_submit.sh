#!/bin/bash
source usr-config.env

# Initialize variables
ARG_IMG=""
ARG_GPU=""
ARG_CMD=""
ARG_ENV=""
JOB_NAME=""
ARG_POOL=""            # <-- new variable for node pool
is_interactive=false
run_jupyter=false

# Function to display help message
function show_help {
    echo "Usage: $0 [-n <image>] [-g <gpu_fraction>] [-c <command>] [-e <env_file>] [-f <name>] [-p <node_pool>] [-i] [-j] [-h]"
    echo "  -n <image>      : Image name"
    echo "  -g <fract>      : GPU fraction/amount"
    echo "  -c <command>    : Command to run (optional)"
    echo "  -e <fpath>      : Path to env.yaml file (optional)"
    echo "  -f <name>       : Custom job name (optional)"
    echo "  -p <node_pool>  : Node pool name (optional)"
    echo "  -i              : Interactive job (optional)"
    echo "  -j              : Jupyter server (optional)"
    echo "  -h              : Help"
}

# Parse flags using getopts
while getopts ":n:g:c:e:f:p:ijh" opt; do
    case $opt in
        n)
            ARG_IMG=$OPTARG
            ;;
        g)
            ARG_GPU=$OPTARG
            ;;
        c)
            ARG_CMD=$OPTARG
            ;;
        e)
            ARG_ENV=$OPTARG
            ;;
        f)
            JOB_NAME=$OPTARG
            ;;
        p)
            ARG_POOL=$OPTARG    # <-- capture node pool
            ;;
        i)
            is_interactive=true
            ;;
        j)
            run_jupyter=true
            ;;
        h)
            show_help
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

INTERACTIVE_FLAG=""
if [ "$is_interactive" = true ]; then
    INTERACTIVE_FLAG="--interactive"
fi

if [ "$run_jupyter" = true ]; then
    ARG_CMD="jupyter lab --port 8888 --no-browser"
    INTERACTIVE_FLAG="--interactive"
fi

if [ -n "$ARG_ENV" ]; then
    ARG_CMD="micromamba create -y -n myenv -f $ARG_ENV && micromamba run -n myenv $ARG_CMD"
fi

# Sanitize command to handle newlines
ARG_CMD=$(echo "$ARG_CMD" | tr '\n' ' ')

# Log the command for debugging
echo "Image [$ARG_IMG], GPU fraction [$ARG_GPU], Command [$ARG_CMD], Node pool [$ARG_POOL]"

if [ -z "$JOB_NAME" ]; then
    # Set job name with image name plus random suffix
    RANDOM_STRING=$((RANDOM % 900000 + 100000))
    ARG_IMG_CLEANED=${ARG_IMG//[^a-zA-Z0-9]/}
    JOB_NAME="${ARG_IMG_CLEANED}-${RANDOM_STRING}"
fi

# Construct the final image name
IMAGE=registry.rcp.epfl.ch/lts2-$LDAP_USERNAME/$ARG_IMG:latest

# Optionally add node-pool if provided
NODE_POOL_FLAG=""
if [ -n "$ARG_POOL" ]; then
    NODE_POOL_FLAG="--node-pool $ARG_POOL"
fi

# Submit job
runai submit \
    $INTERACTIVE_FLAG \
    --large-shm \
    --gpu "$ARG_GPU" \
    --pvc lts2-scratch:/rcp \
    --working-dir "/rcp/${LDAP_USERNAME}" \
    --name "$JOB_NAME" \
    --image "$IMAGE" \
    $NODE_POOL_FLAG \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -- /bin/bash -c "$ARG_CMD"

if [ $? -eq 0 ]; then
	echo ""
    echo "### Check job status ###"
	echo "	runai describe job $JOB_NAME"
    echo "### Delete job ###"
	echo "	runai delete job $JOB_NAME"
    echo "### Get job logs ###"
	echo "	runai logs $JOB_NAME"
	echo "### Connect - terminal ###"
	echo "	runai bash $JOB_NAME"
	echo "### Connect - jupyter ###"
	echo "	runai port-forward $JOB_NAME --port 8888:8888"
	echo "	open in browser: http://localhost:8888"
    echo "  use runai logs to get the token..."
fi
