# Docker Usage for PySimpleMask

## Build the Image

You can build the Docker image using either Docker or Podman.

### Docker
```bash
docker build -t pysimplemask .
```

### Podman
```bash
podman build -t pysimplemask .
```

## Run the Container (Linux)

To run the GUI application from the container, you need to allow it to access your local X11 display.

### Allow X11 Access
Run this command on your host machine to allow local connections to X11:
```bash
xhost +local:
```

### Run with Docker
```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/data \
    pysimplemask
```
*Note: We mount the current directory to `/data` so you can easily access files. You may need to adjust the path argument in the application or pass it via CLI if supported.*

### Run with Podman
Podman often handles rootless containers differently.

```bash
podman run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --security-opt label=type:container_runtime_t \
    -v $(pwd):/data \
    pysimplemask
```
*Note: The `--security-opt label=type:container_runtime_t` might be needed on systems with SELinux (like Fedora/RHEL/CentOS).*

## Troubleshooting

If you see errors related to `libGL` or `xcb` plugin, ensure you have allowed X11 access with `xhost +local:` and that the environment variable `DISPLAY` is passed correctly.
