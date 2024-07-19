from __future__ import annotations

import inspect
import sys
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter
from itertools import count
from keyword import iskeyword
from typing import Any, ClassVar

import sqlalchemy
from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    Column,
    Computed,
    Constraint,
    DefaultClause,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import CompileError
from sqlalchemy.sql.elements import TextClause

from ..models import (
    Model,
    ModelClass,
)
from ..models import Base, LiteralImport
from ..utils import (
    decode_postgresql_sequence,
    get_column_names,
    get_compiled_expression,
    get_constraint_sort_key,
    render_callable,
    uses_default_name,
    re_invalid_identifier,
    re_column_name,
    re_boolean_check_constraint,
    re_enum_check_constraint,
    re_enum_item,
)




class CodeGenerator(metaclass=ABCMeta):
    valid_options: ClassVar[set[str]] = set()

    def __init__(
        self,
        metadata: MetaData,
        bind: Connection | Engine,
        options: Sequence[str],
    ):
        self.metadata: MetaData = metadata
        self.bind: Connection | Engine = bind
        self.options: set[str] = set(options)

        # Validate options
        invalid_options = {opt for opt in options if opt not in self.valid_options}
        if invalid_options:
            raise ValueError("Unrecognized options: " + ", ".join(invalid_options))

    @abstractmethod
    def generate(self) -> str:
        """
        Generate the code for the given metadata.
        .. note:: May modify the metadata.
        """


@dataclass(eq=False)
class TablesGenerator(CodeGenerator):
    valid_options: ClassVar[set[str]] = {
        "noindexes",
        "noconstraints",
        "nocomments",
    }
    builtin_module_names: ClassVar[set[str]] = set(sys.builtin_module_names) | {
        "dataclasses"
    }

    def __init__(
        self,
        metadata: MetaData,
        bind: Connection | Engine,
        options: Sequence[str],
        *,
        indentation: str = "    ",
    ):
        super().__init__(metadata, bind, options)
        self.indentation: str = indentation
        self.imports: dict[str, set[str]] = defaultdict(set)
        self.module_imports: set[str] = set()

    def generate_base(self) -> None:
        self.base = Base(
            literal_imports=[LiteralImport("sqlalchemy", "MetaData")],
            declarations=["metadata = MetaData()"],
            metadata_ref="metadata",
        )

    def generate(self) -> str:
        self.generate_base()

        sections: list[str] = []

        # Remove unwanted elements from the metadata
        for table in list(self.metadata.tables.values()):
            if self.should_ignore_table(table):
                self.metadata.remove(table)
                continue

            if "noindexes" in self.options:
                table.indexes.clear()

            if "noconstraints" in self.options:
                table.constraints.clear()

            if "nocomments" in self.options:
                table.comment = None

            for column in table.columns:
                if "nocomments" in self.options:
                    column.comment = None

        # Use information from column constraints to figure out the intended column
        # types
        for table in self.metadata.tables.values():
            self.fix_column_types(table)

        # Generate the models
        models: list[Model] = self.generate_models()

        # Render module level variables
        variables = self.render_module_variables(models)
        if variables:
            sections.append(variables + "\n")

        # Render models
        self.enums_to_generate = set()
        rendered_models = self.render_models(models)
        if rendered_models:
            sections.append(rendered_models)

        # Render enums
        for enum_name, enum_values in self.enums_to_generate:
            rendered_enum = self.render_python_enum(enum_name, enum_values)
            sections.insert(0, rendered_enum)  # Insert enums at the beginning

        # Render collected imports
        groups = self.group_imports()
        imports = "\n\n".join("\n".join(line for line in group) for group in groups)
        if imports:
            sections.insert(0, imports)

        return "\n\n".join(sections) + "\n"

    def collect_imports(self, models: Iterable[Model]) -> None:
        for literal_import in self.base.literal_imports:
            self.add_literal_import(literal_import.pkgname, literal_import.name)

        for model in models:
            self.collect_imports_for_model(model)

    def collect_imports_for_model(self, model: Model) -> None:
        if model.__class__ is Model:
            self.add_import(Table)

        for column in model.table.c:
            self.collect_imports_for_column(column)

        for constraint in model.table.constraints:
            self.collect_imports_for_constraint(constraint)

        for index in model.table.indexes:
            self.collect_imports_for_constraint(index)

    def collect_imports_for_column(self, column: Column[Any]) -> None:
        self.add_import(column.type)

        if isinstance(column.type, ARRAY):
            self.add_import(column.type.item_type.__class__)
        elif isinstance(column.type, Enum):
            self.add_module_import("enum")
        elif isinstance(column.type, JSONB):
            if (
                not isinstance(column.type.astext_type, Text)
                or column.type.astext_type.length is not None
            ):
                self.add_import(column.type.astext_type)

        if column.default:
            self.add_import(column.default)

        if column.server_default:
            if isinstance(column.server_default, (Computed, Identity)):
                self.add_import(column.server_default)
            elif isinstance(column.server_default, DefaultClause):
                self.add_literal_import("sqlalchemy", "text")

    def collect_imports_for_constraint(self, constraint: Constraint | Index) -> None:
        if isinstance(constraint, Index):
            if len(constraint.columns) > 1 or not uses_default_name(constraint):
                self.add_literal_import("sqlalchemy", "Index")
        elif isinstance(constraint, PrimaryKeyConstraint):
            if not uses_default_name(constraint):
                self.add_literal_import("sqlalchemy", "PrimaryKeyConstraint")
        elif isinstance(constraint, UniqueConstraint):
            if len(constraint.columns) > 1 or not uses_default_name(constraint):
                self.add_literal_import("sqlalchemy", "UniqueConstraint")
        elif isinstance(constraint, ForeignKeyConstraint):
            if len(constraint.columns) > 1 or not uses_default_name(constraint):
                self.add_literal_import("sqlalchemy", "ForeignKeyConstraint")
            else:
                self.add_import(ForeignKey)
        else:
            self.add_import(constraint)

    def add_import(self, obj: Any) -> None:
        # Don't store builtin imports
        if getattr(obj, "__module__", "builtins") == "builtins":
            return

        type_ = type(obj) if not isinstance(obj, type) else obj
        pkgname = type_.__module__

        # The column types have already been adapted towards generic types if possible,
        # so if this is still a vendor specific type (e.g., MySQL INTEGER) be sure to
        # use that rather than the generic sqlalchemy type as it might have different
        # constructor parameters.
        if pkgname.startswith("sqlalchemy.dialects."):
            dialect_pkgname = ".".join(pkgname.split(".")[0:3])
            dialect_pkg = import_module(dialect_pkgname)

            if type_.__name__ in dialect_pkg.__all__:
                pkgname = dialect_pkgname
        elif type_.__name__ in dir(sqlalchemy):
            pkgname = "sqlalchemy"
        else:
            pkgname = type_.__module__

        self.add_literal_import(pkgname, type_.__name__)

    def add_literal_import(self, pkgname: str, name: str) -> None:
        names = self.imports.setdefault(pkgname, set())
        names.add(name)

    def remove_literal_import(self, pkgname: str, name: str) -> None:
        names = self.imports.setdefault(pkgname, set())
        if name in names:
            names.remove(name)

    def add_module_import(self, pgkname: str) -> None:
        self.module_imports.add(pgkname)

    def group_imports(self) -> list[list[str]]:
        future_imports: list[str] = []
        stdlib_imports: list[str] = []
        thirdparty_imports: list[str] = []

        for package in sorted(self.imports):
            imports = ", ".join(sorted(self.imports[package]))
            collection = thirdparty_imports
            if package == "__future__":
                collection = future_imports
            elif package in self.builtin_module_names:
                collection = stdlib_imports
            elif package in sys.modules:
                if "site-packages" not in (sys.modules[package].__file__ or ""):
                    collection = stdlib_imports

            collection.append(f"from {package} import {imports}")

        for module in sorted(self.module_imports):
            thirdparty_imports.append(f"import {module}")

        return [
            group
            for group in (future_imports, stdlib_imports, thirdparty_imports)
            if group
        ]

    def generate_models(self) -> list[Model]:
        models = [Model(table) for table in self.metadata.sorted_tables]

        # Collect the imports
        self.collect_imports(models)

        # Generate names for models
        global_names = {name for namespace in self.imports.values() for name in namespace}
        for model in models:
            self.generate_model_name(model, global_names)
            global_names.add(model.name)

        return models

    def generate_model_name(self, model: Model, global_names: set[str]) -> None:
        preferred_name = f"t_{model.table.name}"
        model.name = self.find_free_name(preferred_name, global_names)

    def render_module_variables(self, models: list[Model]) -> str:
        declarations = self.base.declarations

        if any(not isinstance(model, ModelClass) for model in models):
            if self.base.table_metadata_declaration is not None:
                declarations.append(self.base.table_metadata_declaration)

        return "\n".join(declarations)

    def render_models(self, models: list[Model]) -> str:
        rendered: list[str] = []
        for model in models:
            rendered_table = self.render_table(model.table)
            rendered.append(f"{model.name} = {rendered_table}")

        return "\n\n".join(rendered)

    def render_table(self, table: Table) -> str:
        args: list[str] = [f"{table.name!r}, {self.base.metadata_ref}"]
        kwargs: dict[str, object] = {}
        for column in table.columns:
            # Cast is required because of a bug in the SQLAlchemy stubs regarding
            # Table.columns
            args.append(self.render_column(column, True, is_table=True))

        for constraint in sorted(table.constraints, key=get_constraint_sort_key):
            if uses_default_name(constraint):
                if isinstance(constraint, PrimaryKeyConstraint):
                    continue
                elif isinstance(constraint, (ForeignKeyConstraint, UniqueConstraint)):
                    if len(constraint.columns) == 1:
                        continue

            args.append(self.render_constraint(constraint))

        for index in sorted(table.indexes, key=lambda i: i.name):
            # One-column indexes should be rendered as index=True on columns
            if len(index.columns) > 1 or not uses_default_name(index):
                args.append(self.render_index(index))

        if table.schema:
            kwargs["schema"] = repr(table.schema)

        table_comment = getattr(table, "comment", None)
        if table_comment:
            kwargs["comment"] = repr(table.comment)

        return render_callable("Table", *args, kwargs=kwargs, indentation="    ")

    def render_index(self, index: Index) -> str:
        extra_args = [repr(col.name) for col in index.columns]
        kwargs = {}
        if index.unique:
            kwargs["unique"] = True

        return render_callable("Index", repr(index.name), *extra_args, kwargs=kwargs)

    # TODO find better solution for is_table
    def render_column(
        self, column: Column[Any], show_name: bool, is_table: bool = False
    ) -> str:
        args = []
        kwargs: dict[str, Any] = {}
        kwarg = []
        is_sole_pk = column.primary_key and len(column.table.primary_key) == 1
        dedicated_fks = [
            c
            for c in column.foreign_keys
            if c.constraint
            and len(c.constraint.columns) == 1
            and uses_default_name(c.constraint)
        ]
        is_unique = any(
            isinstance(c, UniqueConstraint)
            and set(c.columns) == {column}
            and uses_default_name(c)
            for c in column.table.constraints
        )
        is_unique = is_unique or any(
            i.unique and set(i.columns) == {column} and uses_default_name(i)
            for i in column.table.indexes
        )
        is_primary = (
            any(
                isinstance(c, PrimaryKeyConstraint)
                and column.name in c.columns
                and uses_default_name(c)
                for c in column.table.constraints
            )
            or column.primary_key
        )
        has_index = any(
            set(i.columns) == {column} and uses_default_name(i)
            for i in column.table.indexes
        )

        if show_name:
            args.append(repr(column.name))

        # Render the column type if there are no foreign keys on it or any of them
        # points back to itself
        if not dedicated_fks or any(fk.column is column for fk in dedicated_fks):
            args.append(self.render_column_type(column.type))

        for fk in dedicated_fks:
            args.append(self.render_constraint(fk))

        if column.default:
            args.append(repr(column.default))

        if column.key != column.name:
            kwargs["key"] = column.key
        if is_primary:
            kwargs["primary_key"] = True
        if not column.nullable and not is_sole_pk and is_table:
            kwargs["nullable"] = False

        if is_unique:
            column.unique = True
            kwargs["unique"] = True
        if has_index:
            column.index = True
            kwarg.append("index")
            kwargs["index"] = True

        if isinstance(column.server_default, DefaultClause):
            kwargs["server_default"] = render_callable(
                "text", repr(column.server_default.arg.text)
            )
        elif isinstance(column.server_default, Computed):
            expression = str(column.server_default.sqltext)

            computed_kwargs = {}
            if column.server_default.persisted is not None:
                computed_kwargs["persisted"] = column.server_default.persisted

            args.append(
                render_callable("Computed", repr(expression), kwargs=computed_kwargs)
            )
        elif isinstance(column.server_default, Identity):
            args.append(repr(column.server_default))
        elif column.server_default:
            kwargs["server_default"] = repr(column.server_default)

        comment = getattr(column, "comment", None)
        if comment:
            kwargs["comment"] = repr(comment)

        if is_table:
            self.add_import(Column)
            return render_callable("Column", *args, kwargs=kwargs)
        else:
            return render_callable("mapped_column", *args, kwargs=kwargs)

    def render_column_type(self, coltype: object) -> str:
        args = []
        kwargs: dict[str, Any] = {}
        sig = inspect.signature(coltype.__class__.__init__)
        defaults = {param.name: param.default for param in sig.parameters.values()}
        missing = object()
        use_kwargs = False
        for param in list(sig.parameters.values())[1:]:
            # Remove annoyances like _warn_on_bytestring
            if param.name.startswith("_"):
                continue
            elif param.kind in (
                Parameter.VAR_POSITIONAL,
                Parameter.VAR_KEYWORD,
            ):
                continue

            value = getattr(coltype, param.name, missing)
            default = defaults.get(param.name, missing)
            if value is missing or value == default:
                use_kwargs = True
            elif use_kwargs:
                kwargs[param.name] = repr(value)
            else:
                args.append(repr(value))

        vararg = next(
            (
                param.name
                for param in sig.parameters.values()
                if param.kind is Parameter.VAR_POSITIONAL
            ),
            None,
        )
        if vararg and hasattr(coltype, vararg):
            varargs_repr = [repr(arg) for arg in getattr(coltype, vararg)]
            args.extend(varargs_repr)

        if isinstance(coltype, Enum) and coltype.name is not None:
            # Generate a Python-style class name from the Enum name
            enum_class_name = "".join(
                word.capitalize() for word in coltype.name.split("_")
            )
            # Add the enum class name and its values to the set
            self.enums_to_generate.add((enum_class_name, tuple(coltype.enums)))
            # Clear existing args and kwargs, then add only the enum class name
            args = [enum_class_name]
            kwargs = {}

        if isinstance(coltype, JSONB):
            # Remove astext_type if it's the default
            if (
                isinstance(coltype.astext_type, Text)
                and coltype.astext_type.length is None
            ):
                del kwargs["astext_type"]

        if args or kwargs:
            return render_callable(coltype.__class__.__name__, *args, kwargs=kwargs)
        else:
            return coltype.__class__.__name__

    def render_constraint(self, constraint: Constraint | ForeignKey) -> str:
        def add_fk_options(*opts: Any) -> None:
            args.extend(repr(opt) for opt in opts)
            for attr in (
                "ondelete",
                "onupdate",
                "deferrable",
                "initially",
                "match",
            ):
                value = getattr(constraint, attr, None)
                if value:
                    kwargs[attr] = repr(value)

        args: list[str] = []
        kwargs: dict[str, Any] = {}
        if isinstance(constraint, ForeignKey):
            remote_column = f"{constraint.column.table.fullname}.{constraint.column.name}"
            add_fk_options(remote_column)
        elif isinstance(constraint, ForeignKeyConstraint):
            local_columns = get_column_names(constraint)
            remote_columns = [
                f"{fk.column.table.fullname}.{fk.column.name}"
                for fk in constraint.elements
            ]
            add_fk_options(local_columns, remote_columns)
        elif isinstance(constraint, CheckConstraint):
            args.append(repr(get_compiled_expression(constraint.sqltext, self.bind)))
        elif isinstance(constraint, (UniqueConstraint, PrimaryKeyConstraint)):
            args.extend(repr(col.name) for col in constraint.columns)
        else:
            raise TypeError(
                f"Cannot render constraint of type {constraint.__class__.__name__}"
            )

        if isinstance(constraint, Constraint) and not uses_default_name(constraint):
            kwargs["name"] = repr(constraint.name)

        return render_callable(constraint.__class__.__name__, *args, kwargs=kwargs)

    def render_python_enum(self, name: str, values: list[str]) -> str:
        enum_members = "\n    ".join([f"{value.upper()} = '{value}'" for value in values])
        return f"class {name}(enum.Enum):\n    {enum_members}\n"

    def should_ignore_table(self, table: Table) -> bool:
        # Support for Alembic and sqlalchemy-migrate -- never expose the schema version
        # tables
        return table.name in ("alembic_version", "migrate_version")

    def find_free_name(
        self,
        name: str,
        global_names: set[str],
        local_names: Collection[str] = (),
    ) -> str:
        """
        Generate an attribute name that does not clash with other local or global names.
        """
        name = name.strip()
        assert name, "Identifier cannot be empty"
        name = re_invalid_identifier.sub("_", name)
        if name[0].isdigit():
            name = "_" + name
        elif iskeyword(name) or name == "metadata":
            name += "_"

        original = name
        for i in count():
            if name not in global_names and name not in local_names:
                break

            name = original + (str(i) if i else "_")

        return name

    def fix_column_types(self, table: Table) -> None:
        """Adjust the reflected column types."""
        # Detect check constraints for boolean and enum columns
        for constraint in table.constraints.copy():
            if isinstance(constraint, CheckConstraint):
                sqltext = get_compiled_expression(constraint.sqltext, self.bind)

                # Turn any integer-like column with a CheckConstraint like
                # "column IN (0, 1)" into a Boolean
                match = re_boolean_check_constraint.match(sqltext)
                if match:
                    colname_match = re_column_name.match(match.group(1))
                    if colname_match:
                        colname = colname_match.group(3)
                        table.constraints.remove(constraint)
                        table.c[colname].type = Boolean()
                        continue

                # Turn any string-type column with a CheckConstraint like
                # "column IN (...)" into an Enum
                match = re_enum_check_constraint.match(sqltext)
                if match:
                    colname_match = re_column_name.match(match.group(1))
                    if colname_match:
                        colname = colname_match.group(3)
                        items = match.group(2)
                        if isinstance(table.c[colname].type, String):
                            table.constraints.remove(constraint)
                            if not isinstance(table.c[colname].type, Enum):
                                options = re_enum_item.findall(items)
                                table.c[colname].type = Enum(*options, native_enum=False)

                            continue

        for column in table.c:
            try:
                column.type = self.get_adapted_type(column.type)
            except CompileError:
                pass

            # PostgreSQL specific fix: detect sequences from server_default
            if column.server_default and self.bind.dialect.name == "postgresql":
                if isinstance(column.server_default, DefaultClause) and isinstance(
                    column.server_default.arg, TextClause
                ):
                    schema, seqname = decode_postgresql_sequence(
                        column.server_default.arg
                    )
                    if seqname:
                        # Add an explicit sequence
                        if seqname != f"{column.table.name}_{column.name}_seq":
                            column.default = sqlalchemy.Sequence(seqname, schema=schema)

                        column.server_default = None

    def get_adapted_type(self, coltype: Any) -> Any:
        compiled_type = coltype.compile(self.bind.engine.dialect)
        for supercls in coltype.__class__.__mro__:
            if not supercls.__name__.startswith("_") and hasattr(
                supercls, "__visit_name__"
            ):
                # Hack to fix adaptation of the Enum class which is broken since
                # SQLAlchemy 1.2
                kw = {}
                if supercls is Enum:
                    kw["name"] = coltype.name

                try:
                    new_coltype = coltype.adapt(supercls)
                except TypeError:
                    # If the adaptation fails, don't try again
                    break

                for key, value in kw.items():
                    setattr(new_coltype, key, value)

                if isinstance(coltype, ARRAY):
                    new_coltype.item_type = self.get_adapted_type(new_coltype.item_type)

                try:
                    # If the adapted column type does not render the same as the
                    # original, don't substitute it
                    if new_coltype.compile(self.bind.engine.dialect) != compiled_type:
                        # Make an exception to the rule for Float and arrays of Float,
                        # since at least on PostgreSQL, Float can accurately represent
                        # both REAL and DOUBLE_PRECISION
                        if not isinstance(new_coltype, Float) and not (
                            isinstance(new_coltype, ARRAY)
                            and isinstance(new_coltype.item_type, Float)
                        ):
                            break
                except CompileError:
                    # If the adapted column type can't be compiled, don't substitute it
                    break

                # Stop on the first valid non-uppercase column type class
                coltype = new_coltype
                if supercls.__name__ != supercls.__name__.upper():
                    break

        return coltype
