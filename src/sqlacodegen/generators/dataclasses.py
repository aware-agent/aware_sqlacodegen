from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.engine import Connection, Engine
from sqlalchemy import MetaData

from ..models import Base, LiteralImport
from .declarative import DeclarativeGenerator


class DataclassGenerator(DeclarativeGenerator):
    def __init__(
        self,
        metadata: MetaData,
        bind: Connection | Engine,
        options: Sequence[str],
        *,
        indentation: str = "    ",
        base_class_name: str = "Base",
        quote_annotations: bool = False,
        metadata_key: str = "sa",
    ):
        super().__init__(
            metadata,
            bind,
            options,
            indentation=indentation,
            base_class_name=base_class_name,
        )
        self.metadata_key: str = metadata_key
        self.quote_annotations: bool = quote_annotations

    def generate_base(self) -> None:
        self.base = Base(
            literal_imports=[
                LiteralImport("sqlalchemy.orm", "DeclarativeBase"),
                LiteralImport("sqlalchemy.orm", "MappedAsDataclass"),
            ],
            declarations=[
                f"class {self.base_class_name}(MappedAsDataclass, DeclarativeBase):",
                f"{self.indentation}pass",
            ],
            metadata_ref=f"{self.base_class_name}.metadata",
        )
