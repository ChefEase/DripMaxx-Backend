from loguru import logger


def setup_logging():
  # Minimal Loguru config; stdout only, structured enough for dev
  logger.remove()
  logger.add(
    sink=lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
  )
  logger.debug("logging configured")
  return logger
