"""value comparison, display, and mutation rules for explicitly initialized records."""

missing = object()


class mutable_record:
    __match_args__: tuple[str, ...] = ()
    __hash__ = None

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return tuple(getattr(self, name) for name in self.__match_args__) == tuple(
            getattr(other, name) for name in self.__match_args__
        )

    def __repr__(self) -> str:
        values = ", ".join(
            f"{name}={getattr(self, name)!r}" for name in self.__match_args__
        )
        return f"{type(self).__qualname__}({values})"


class immutable_record(mutable_record):
    def __setattr__(self, name, value) -> None:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name) -> None:
        raise AttributeError(f"cannot delete field {name!r}")

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, name) for name in self.__match_args__))
