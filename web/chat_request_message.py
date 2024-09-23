from dataclasses import dataclass


@dataclass
class ChatRequestMessage:
    context: str
