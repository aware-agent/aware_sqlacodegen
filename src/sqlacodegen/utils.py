from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from sqlalchemy import PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.schema import (
    CheckConstraint,
    ColumnCollectionConstraint,
    Constraint,
    ForeignKeyConstraint,
    Index,
    Table,
)

# public
re_boolean_check_constraint = re.compile(r"(?:.*?\.)?(.*?) IN \(0, 1\)")
re_column_name = re.compile(r'(?:(["`]?).*\1\.)?(["`]?)(.*)\2')
re_enum_check_constraint = re.compile(r"(?:.*?\.)?(.*?) IN \((.+)\)")
re_enum_item = re.compile(r"'(.*?)(?<!\\)'")
re_invalid_identifier = re.compile(r"(?u)\W")

# private
_re_postgresql_nextval_sequence = re.compile(r"nextval\('(.+)'::regclass\)")
_re_postgresql_sequence_delimiter = re.compile(r'(.*?)([."]|$)')


def get_column_names(constraint: ColumnCollectionConstraint) -> list[str]:
    return list(constraint.columns.keys())


def get_constraint_sort_key(constraint: Constraint) -> str:
    if isinstance(constraint, CheckConstraint):
        return f"C{constraint.sqltext}"
    elif isinstance(constraint, ColumnCollectionConstraint):
        return constraint.__class__.__name__[0] + repr(get_column_names(constraint))
    else:
        return str(constraint)


def get_compiled_expression(statement: ClauseElement, bind: Engine | Connection) -> str:
    """Return the statement in a form where any placeholders have been filled in."""
    return str(statement.compile(bind, compile_kwargs={"literal_binds": True}))


def get_common_fk_constraints(
    table1: Table, table2: Table
) -> set[ForeignKeyConstraint]:
    """
    Return a set of foreign key constraints the two tables have against each other.

    """
    c1 = {
        c
        for c in table1.constraints
        if isinstance(c, ForeignKeyConstraint) and c.elements[0].column.table == table2
    }
    c2 = {
        c
        for c in table2.constraints
        if isinstance(c, ForeignKeyConstraint) and c.elements[0].column.table == table1
    }
    return c1.union(c2)


def uses_default_name(constraint: Constraint | Index) -> bool:
    if not constraint.name or constraint.table is None:
        return True

    table = constraint.table
    values: dict[str, Any] = {
        "table_name": table.name,
        "constraint_name": constraint.name,
    }
    if isinstance(constraint, (Index, ColumnCollectionConstraint)):
        values.update(
            {
                "column_0N_name": "".join(col.name for col in constraint.columns),
                "column_0_N_name": "_".join(col.name for col in constraint.columns),
                "column_0N_label": "".join(
                    col.label(col.name).name for col in constraint.columns
                ),
                "column_0_N_label": "_".join(
                    col.label(col.name).name for col in constraint.columns
                ),
                "column_0N_key": "".join(
                    col.key for col in constraint.columns if col.key
                ),
                "column_0_N_key": "_".join(
                    col.key for col in constraint.columns if col.key
                ),
            }
        )
        if constraint.columns:
            columns = constraint.columns.values()
            values.update(
                {
                    "column_0_name": columns[0].name,
                    "column_0_label": columns[0].label(columns[0].name).name,
                    "column_0_key": columns[0].key,
                }
            )

    if isinstance(constraint, Index):
        key = "ix"
    elif isinstance(constraint, CheckConstraint):
        key = "ck"
    elif isinstance(constraint, UniqueConstraint):
        key = "uq"
    elif isinstance(constraint, PrimaryKeyConstraint):
        key = "pk"
    elif isinstance(constraint, ForeignKeyConstraint):
        key = "fk"
        values.update(
            {
                "referred_table_name": constraint.referred_table,
                "referred_column_0_name": constraint.elements[0].column.name,
                "referred_column_0N_name": "".join(
                    fk.column.name for fk in constraint.elements
                ),
                "referred_column_0_N_name": "_".join(
                    fk.column.name for fk in constraint.elements
                ),
                "referred_column_0_label": constraint.elements[0]
                .column.label(constraint.elements[0].column.name)
                .name,
                "referred_fk.column_0N_label": "".join(
                    fk.column.label(fk.column.name).name for fk in constraint.elements
                ),
                "referred_fk.column_0_N_label": "_".join(
                    fk.column.label(fk.column.name).name for fk in constraint.elements
                ),
                "referred_fk.column_0_key": constraint.elements[0].column.key,
                "referred_fk.column_0N_key": "".join(
                    fk.column.key for fk in constraint.elements if fk.column.key
                ),
                "referred_fk.column_0_N_key": "_".join(
                    fk.column.key for fk in constraint.elements if fk.column.key
                ),
            }
        )
    else:
        raise TypeError(f"Unknown constraint type: {constraint.__class__.__qualname__}")

    try:
        convention: str = table.metadata.naming_convention[key]
        return constraint.name == (convention % values)
    except KeyError:
        return False


def render_callable(
    name: str,
    *args: object,
    kwargs: Mapping[str, object] | None = None,
    indentation: str = "",
) -> str:
    """
    Render a function call.

    :param name: name of the callable
    :param args: positional arguments
    :param kwargs: keyword arguments
    :param indentation: if given, each argument will be rendered on its own line with
        this value used as the indentation

    """
    if kwargs:
        args += tuple(f"{key}={value}" for key, value in kwargs.items())

    if indentation:
        prefix = f"\n{indentation}"
        suffix = "\n"
        delimiter = f",\n{indentation}"
    else:
        prefix = suffix = ""
        delimiter = ", "

    rendered_args = delimiter.join(str(arg) for arg in args)

    return f"{name}({prefix}{rendered_args}{suffix})"


def qualified_table_name(table: Table) -> str:
    if table.schema:
        return f"{table.schema}.{table.name}"
    else:
        return str(table.name)


def decode_postgresql_sequence(clause: TextClause) -> tuple[str | None, str | None]:
    match = _re_postgresql_nextval_sequence.match(clause.text)
    if not match:
        return None, None

    schema: str | None = None
    sequence: str = ""
    in_quotes = False
    for match in _re_postgresql_sequence_delimiter.finditer(match.group(1)):
        sequence += match.group(1)
        if match.group(2) == '"':
            in_quotes = not in_quotes
        elif match.group(2) == ".":
            if in_quotes:
                sequence += "."
            else:
                schema, sequence = sequence, ""

    return schema, sequence


def get_python_type(sql_type: str, is_array: bool = False) -> str:
    """
    Map PostgreSQL types to Python types.
    """
    type_mapping = {
        # Numeric types
        "smallint": "int",
        "integer": "int",
        "bigint": "int",
        "decimal": "Decimal",
        "numeric": "Decimal",
        "real": "float",
        "double precision": "float",
        "serial": "int",
        "bigserial": "int",
        # Monetary types
        "money": "str",
        # Character types
        "character varying": "str",
        "varchar": "str",
        "character": "str",
        "char": "str",
        "text": "str",
        # Binary Data Types
        "bytea": "bytes",
        # Date/Time Types
        "timestamp without time zone": "datetime",
        "timestamp with time zone": "datetime",
        "date": "date",
        "time without time zone": "time",
        "time with time zone": "time",
        "interval": "timedelta",
        # Boolean Type
        "boolean": "bool",
        # Enum Type
        "enum": "Enum",  # Custom handling may be needed for specific enums
        # Geometric Types
        "point": "tuple",
        "line": "str",
        "lseg": "tuple",
        "box": "tuple",
        "path": "list",
        "polygon": "list",
        "circle": "tuple",
        # Network Address Types
        "cidr": "str",
        "inet": "str",
        "macaddr": "str",
        # Bit String Types
        "bit": "str",
        "bit varying": "str",
        # UUID Type
        "UUID": "str", # TODO use UUID!
        "uuid": "str", # TODO use UUID!
        # JSON Types
        "json": "dict",
        "jsonb": "dict",
        # XML Type
        "xml": "str",
        # Full Text Search Types
        "tsvector": "str",
        "tsquery": "str",
        # Other/miscellaneous types
        "oid": "int",
        "range": "range",  # Custom handling might be needed
        # Void Type
        "void": "None",
    }

    is_array, is_set, base_type = _is_array_or_set(sql_type)

    python_type = type_mapping.get(base_type, "Any")

    if is_array:
        return f"List[{python_type}]"
    if is_set:
        return f"Set[{python_type}]"

    return python_type
    # # Array handling
    # python_type = type_mapping.get(sql_type, "Any")
    # return f"List[{python_type}]" if is_array else python_type


def _is_array_or_set(sql_type: str) -> tuple[bool, bool, str]:
    """
    Determine if the SQL type is an array or set and return the base type.
    """
    sql_type = sql_type.lower()  # Convert to lowercase for easier comparison
    is_array = sql_type.endswith("[]")
    is_set = sql_type.startswith("setof ")

    base_type = sql_type
    if is_array:
        base_type = sql_type[:-2]
    elif is_set:
        base_type = sql_type[6:]

    return is_array, is_set, base_type
