#!/bin/bash

# Define image name
IMAGE_NAME="pysimplemask"

# Check for container runtime (podman or docker)
if command -v podman &> /dev/null; then
    RUNTIME="podman"
    # SELinux workarounds for Podman
    OPTS="--security-opt label=type:container_runtime_t"
elif command -v docker &> /dev/null; then
    RUNTIME="docker"
    OPTS=""
else
    echo "Error: Neither podman nor docker found."
    exit 1
fi

echo "Using runtime: $RUNTIME"

# Build the image if it doesn't exist
IMAGE_EXISTS=false
if [ "$RUNTIME" == "podman" ]; then
    if $RUNTIME image exists $IMAGE_NAME; then
        IMAGE_EXISTS=true
    fi
else
    if $RUNTIME image inspect $IMAGE_NAME > /dev/null 2>&1; then
        IMAGE_EXISTS=true
    fi
fi

if [ "$IMAGE_EXISTS" = false ]; then
    echo "Building image $IMAGE_NAME..."
    $RUNTIME build -t $IMAGE_NAME .
    if [ $? -ne 0 ]; then
        echo "Build failed."
        exit 1
    fi
fi

# Allow X11 connections
echo "Allowing local X11 connections..."
xhost +local:

# Run the container
echo "Starting $IMAGE_NAME..."
$RUNTIME run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/data \
    --net=host \
    $OPTS \
    $IMAGE_NAME

