from .perception_module_runner import PerceptionModuleRunner
import logging
import argparse
import signal

perception = None


def stop_all_threads():
    print("Stopping threads")
    if perception is not None:
        perception.stop()


def shutdown(a, b):
    print("Shutdown was called")
    stop_all_threads()
    exit(0)


def parse_args():
    parser = argparse.ArgumentParser("Perception SystemRunner module")
    parser.add_argument('model_cfg')
    parser.add_argument('weights_path')

    return parser.parse_args()


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    file_handler = logging.handlers.WatchedFileHandler("perception.log", 'w')
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(ch)
    logger.addHandler(file_handler)


def main():
    for signame in ('SIGINT', 'SIGTERM'):
        signal.signal(getattr(signal, signame), shutdown)

    args = parse_args()
    setup_logger()

    logging.info('Starting perception module')
    perception = PerceptionModuleRunner(args.weights_path, args.model_cfg)

    perception.start()
    perception.run()

    stop_all_threads()
    exit(0)

if __name__ == "__main__":
    main()