# coding: utf-8
import os
import sys
from PIL import Image
import numpy as np

from pyFormulaClientNoNvidia import messages
from pyFormulaClientNoNvidia.FormulaClient import FormulaClient, ClientSource, SYSTEM_RUNNER_IPC_PORT

def save_camera_msg(msg, dst_dir):
    camera_data = messages.sensors.CameraSensor()
    msg.data.Unpack(camera_data)
    pixels = camera_data.pixels
    out_file = os.path.join(dst_dir, f'img_{msg.header.id}.png')
    with Image.frombytes("RGB", (camera_data.width, camera_data.height), pixels, 'raw', 'RGBX', 0, -1) as img:
        img.save(out_file, format="PNG")


def main(msg_file, dst_dir): 
    client = FormulaClient(ClientSource.SERVER, 
        read_from_file=msg_file, write_to_file=os.devnull) # the output file 
    conn = client.connect(SYSTEM_RUNNER_IPC_PORT)
    conn = client.connect(SYSTEM_RUNNER_IPC_PORT)
    msg = messages.common.Message()
    while not msg.data.Is(messages.server.ExitMessage.DESCRIPTOR):
        msg = conn.read_message()
        if msg.data.Is(messages.sensors.CameraSensor.DESCRIPTOR):
            save_camera_msg(msg, dst_dir)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("print_messages_file.py <message file> <dest dir>")
        exit(1)
    main(sys.argv[1], sys.argv[2])