# Advanced Flower Example (PyTorch)

This example demonstrates an advanced federated learning setup using Flower with PyTorch. It differs from the quickstart example in the following ways:

- 10 clients (instead of just 2)
- Each client holds a local dataset of 5000 training examples and 1000 test examples
- Server-side model evaluation after parameter aggregation
- Hyperparameter schedule using config functions
- Custom return values
- Server-side parameter initialization


# Run Federated Learning with PyTorch and Flower

The included `run.sh` will start the Flower server (using `server.py`), sleep for 2 seconds to ensure the the server is up, and then start 10 Flower clients (using `client.py`). You can simply start everything in a terminal as follows:

```shell
poetry run ./run.sh
```

The `run.sh` script starts processes in the background so that you don't have to open eleven terminal windows. If you experiment with the code example and something goes wrong, simply using `CTRL + C` on Linux (or `CMD + C` on macOS) wouldn't normally kill all these processes, which is why the script ends with `trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT` and `wait`. This simply allows you to stop the experiment using `CTRL + C` (or `CMD + C`). If you change the script and anyhting goes wrong you can still use `killall python` (or `killall python3`) to kill all background processes (or a more specific command if you have other Python processes running that you don't want to kill).
