"""
Zulu-Walker Framework — Generic entry point.

Usage:
    python run.py                    # Run framework demo with mock platform
    python run.py --platform linux   # Run with Linux platform
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Zulu-Walker Framework")
    parser.add_argument(
        "--platform", default="mock",
        choices=["mock", "linux", "maixcam2"],
        help="Target hardware platform (default: mock)",
    )
    parser.add_argument("--config", default="project_config.yaml",
                        help="Path to config YAML (default: project_config.yaml)")
    parser.add_argument("--demo", action="store_true", default=True,
                        help="Run framework demo (default: True)")
    args = parser.parse_args()

    from framework.event_bus import EventBus
    from framework.hal import Machine
    from framework.module_manager import ModuleManager

    bus = EventBus()

    config_path = args.config
    if not os.path.exists(config_path):
        logging.warning("Config %s not found, using platform=%s defaults", config_path, args.platform)
        config_path = None

    machine = Machine.create(config_path or "project_config.yaml")
    manager = ModuleManager(machine, event_bus=bus)

    logging.info("Zulu-Walker Framework initialized (platform=%s)", args.platform)
    logging.info("Available components: EventBus, StateMachine, HAL, ModuleManager")

    manager.run_main_loop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    except Exception as e:
        logging.error("Unhandled exception: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
