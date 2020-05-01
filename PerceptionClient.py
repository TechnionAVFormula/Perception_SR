from config import CONFIG, IN_MESSAGE_FILE, OUT_MESSAGE_FILE
from config import ConfigEnum

import logging

if (CONFIG  == ConfigEnum.REAL_TIME) or (CONFIG == ConfigEnum.COGNATA_SIMULATION):
    from pyFormulaClient import FormulaClient, messages  
    from pyFormulaClient.ModuleClient import ModuleClient
    from pyFormulaClient.MessageDeque import MessageDeque
elif ( CONFIG == ConfigEnum.LOCAL_TEST):
    from pyFormulaClientNoNvidia import FormulaClient, messages  
    from pyFormulaClientNoNvidia.ModuleClient import ModuleClient
    from pyFormulaClientNoNvidia.MessageDeque import MessageDeque
else:
    raise NameError('User Should Choose Configuration from config.py')


class PerceptionClient(ModuleClient):
    def __init__(self):
        if (CONFIG  == ConfigEnum.REAL_TIME) or (CONFIG == ConfigEnum.COGNATA_SIMULATION):
            super().__init__(FormulaClient.ClientSource.PERCEPTION)       
        elif ( CONFIG == ConfigEnum.LOCAL_TEST):
            super().__init__(FormulaClient.ClientSource.PERCEPTION, IN_MESSAGE_FILE, OUT_MESSAGE_FILE)  
        self.server_messages = MessageDeque()                                              
        self.camera_messages = MessageDeque(maxlen=1)        
        self.depth_camera_messages = MessageDeque(maxlen=1)        
        self.ground_truth = MessageDeque(maxlen=1)        

    def _callback(self, msg):  
        source = FormulaClient.ClientSource(msg.header.source)
        logging.debug(f'Got message ({msg.header.id}){msg.data.TypeName()}  from  {source.name}')
        if msg.data.Is(messages.sensors.CameraSensor.DESCRIPTOR):
            self.camera_messages.put(msg)
        elif msg.data.Is(messages.sensors.DepthCameraSensor.DESCRIPTOR):
            self.depth_camera_messages.put(msg)
        elif msg.data.Is(messages.ground_truth.GroundTruth.DESCRIPTOR):
            self.ground_truth.put(msg)
        else:
            self.server_messages.put(msg)

    def send_message(self, msg, timeout=FormulaClient.DW_TIMEOUT_INFINITE):
        logging.info(f'Sending message ({msg.header.id}){msg.data.TypeName()}')        
        return super().send_message(msg, timeout=timeout)

    def get_camera_message(self, blocking=True, timeout=None):
        return self.camera_messages.get(blocking, timeout), self.depth_camera_messages.get(blocking, timeout), self.ground_truth.get(blocking, timeout)

    def pop_server_message(self, blocking=False, timeout=None):
        return self.server_messages.get(blocking, timeout)
