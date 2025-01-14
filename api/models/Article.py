from pydantic import BaseModel
from typing import List

class ArticlesInput(BaseModel):
    texts: List[str] 