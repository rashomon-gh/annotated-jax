from loguru import logger
from jax import extend




def log_device():
    device = extend.backend.get_backend().platform

    assert device in ["cpu", "gpu", "tpu", "mps"], logger.error(f"Unknown device: {device}")

    logger.info(f"XLA DEVICE: {device}")


if __name__ == "__main__":
    log_device()
