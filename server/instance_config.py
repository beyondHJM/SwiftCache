from dataclasses import dataclass

@dataclass
class ServerConfig:
    host:str = 'localhost'
    port:int = 8000


@dataclass
class InstanceConfig:
    server_config:ServerConfig
    engine_config: dict