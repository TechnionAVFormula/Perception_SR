from enum import Enum

class ConfigEnum(Enum):
    COGNATA_SIMULATION = 1
    REAL_TIME = 2
    LOCAL_TEST = 3



## Choose method of running the whole State Estimation Module
# CONFIG = ConfigEnum.LOCAL_TEST
IN_MESSAGE_FILE = 'Messages/sensors.messages'
OUT_MESSAGE_FILE = 'Messages/perception.messages'

## all options:
CONFIG = ConfigEnum.COGNATA_SIMULATION
# CONFIG = ConfigEnum.LOCAL_TEST 
# CONFIG = ConfigEnum.REAL_TIME

SAVE_RUN_DIR=None
