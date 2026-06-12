from pydantic import BaseModel
from typing import List



class ChatRequest(BaseModel):
    user_id: int
    session_id: str
    message: str