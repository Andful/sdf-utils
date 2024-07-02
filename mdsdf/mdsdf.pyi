from typing import TypeVar

E = TypeVar('E') 

class Hsdf2D[E]:
    def actors() -> list[tuple[E, tuple[int, int]]]:
        pass

    def channels() -> tuple[tuple[E, tuple[int, int]], tuple[E, tuple[int, int]]]:
        pass

    def dot() -> str:
        pass

class Sdf2D[E]:
    def __init__(elements: set[E]) -> None:
        """
        todo
        """
        
    def actors(self) -> list[E]:
        pass

    def add_channel(
        self,
        source: E,
        target: E,
        production_rate: tuple[int, int],
        consumption_rate: tuple[int, int],
        initial_tokens: tuple[int, int],
    ) -> None:
        pass

    def cut_channel(
        channel: tuple[E, E],
        new_actor: E,
    ):
        pass

    def channels(self) -> list[tuple[E, E]]:
        pass

    def induce(self, elems: set[E]) -> Sdf2D[E]:
        pass
        
    def dot(self) -> str:
        pass

    def hsdf(self) -> Hsdf2D:
        pass

    def nodes(self) -> list[E]:
        pass
