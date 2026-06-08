import zmq
import json
import os
from typing import Dict, Any, Optional, List
import time

DEFAULT_ADDRESS = "ipc:///tmp/zmq_default.sock"




class ZMQClient:
    def __init__(self, name: str = "", address: str = DEFAULT_ADDRESS, socket_type=zmq.DEALER):
        self.context = zmq.Context()
        self.name = name
        self.socket = self.context.socket(socket_type)
        self.socket.setsockopt(zmq.IDENTITY, self.name.encode('utf-8'))
        self.socket.connect(address)

    def send_dict(self, data: Dict[str, Any]):
        data_bytes = json.dumps(data).encode('utf-8')
        self.socket.send(data_bytes)

    def recv_dict_nonblock(self) -> Optional[Dict[str, Any]]:
        try:
            data_bytes = self.socket.recv(flags=zmq.NOBLOCK)
            return json.loads(data_bytes.decode('utf-8'))
        except zmq.Again:
            return None

    def recv_dict(self) -> Dict[str, Any]:
        """接收一个字典"""
        data_bytes = self.socket.recv()
        return json.loads(data_bytes.decode('utf-8'))


    def close(self):
        self.socket.close()
        self.context.term()
    

class ZMQServer:
    def __init__(self, address: str = DEFAULT_ADDRESS, socket_type=zmq.ROUTER):
        self.context = zmq.Context()
        self.socket = self.context.socket(socket_type)
        if address.startswith("ipc://"):
            path = address.replace("ipc://", "")
            if os.path.exists(path):
                os.remove(path)
        self.socket.bind(address)

    def recv_dict_nonblock(self) -> Optional[tuple[str, Dict[str, Any]]]:
        """非阻塞收一条消息，返回(client_id, dict)"""
        try:
            ident, data_bytes = self.socket.recv_multipart(flags=zmq.NOBLOCK)
            return ident, json.loads(data_bytes.decode('utf-8'))
        except zmq.Again:
            return None

    def recv_all_dict_nonblock(self) -> List[tuple[str, Dict[str, Any]]]:
        messages = []
        while True:
            try:
                ident, data_bytes = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                messages.append((ident, json.loads(data_bytes.decode('utf-8'))))
            except zmq.Again:
                break
        return messages

    def send_dict(self, ident: bytes, data: Dict[str, Any]):
        """发送字典给指定客户端"""
        data_bytes = json.dumps(data).encode('utf-8')
        self.socket.send_multipart([ident, data_bytes])
    
    def recv_dict(self) -> tuple[bytes, Dict[str, Any]]:
        """阻塞接收一条消息"""
        ident, data_bytes = self.socket.recv_multipart()  # 阻塞直到收到数据
        return ident, json.loads(data_bytes.decode('utf-8'))

    def close(self):
        self.socket.close()
        self.context.term()
