#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python3 -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

# Download the EfficientNetB0 model
python3 -c "import torch; torch.hub.load( \
        'NVIDIA/DeepLearningExamples:torchhub', \
        'nvidia_efficientnet_b0', pretrained=True)"

#python3 server.py &
#sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 2`; do
    echo "Starting client $i"
    python3 client.py --partition=${i} &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
