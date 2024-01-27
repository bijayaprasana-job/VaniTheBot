from src.framework.Handler import Handler
from abc import abstractmethod
from typing import Any, Optional

from src.utils.CommonUtils import CommonUtils


class AbstractHandler(Handler):
    _next_handler: Handler = None

    def set_next(self, handler: Handler) -> Handler:
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self,*args) -> str:
        if self._next_handler:
            return self._next_handler.handle(*args)
        return None

