from typing import Iterable

from ..models import Model, LiteralImport

Imports = dict[str, set[str]] # type alias

def add_literal_import(imports : Imports, pkgname: str, name: str) -> None:
    names = imports.setdefault(pkgname, set())
    names.add(name)

def remove_literal_import(self, pkgname: str, name: str) -> None:
    names = self.imports.setdefault(pkgname, set())
    if name in names:
        names.remove(name)

def add_module_import(self, pgkname: str) -> None:
    self.module_imports.add(pgkname)

