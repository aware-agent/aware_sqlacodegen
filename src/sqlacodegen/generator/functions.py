from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import (
    Result,
    create_engine,
    text,
)


@dataclass
class ArgumentInfo:
    name: str
    type: str
    default: Any = None
    class_sourced: bool = False
    class_attribute: str = ""


@dataclass
class FunctionMetadata:
    schema: str
    name: str
    return_type: str
    arguments: list[ArgumentInfo]
    docstring: str = ""
    function_table: str = ""
    function_name: str = ""
    function_type: str = "free"
    function_class_values: dict[str, str] = field(default_factory=dict)


def fetch_functions(engine_url) -> dict[str, list[FunctionMetadata]]:
    engine = create_engine(engine_url)
    query = text(_get_function_query())
    with engine.connect() as conn:
        result = conn.execute(query)
        return _process_query_results(result)


def _get_function_query() -> str:
    return """
        SELECT n.nspname as schema,
            p.proname as name,
            pg_catalog.pg_get_function_result(p.oid) as return_type,
            pg_catalog.pg_get_function_arguments(p.oid) as argument_types,
            obj_description(p.oid, 'pg_proc') as function_comment
        FROM pg_catalog.pg_proc p
        LEFT JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
            AND p.prokind = 'f'
        ORDER BY 1, 2;
    """


def _process_query_results(result: Result) -> dict[str, list[FunctionMetadata]]:
    class_functions: dict[str, list[FunctionMetadata]] = {}
    for row in result.fetchall():
        docstring, metadata = _parse_comments(row[4])
        function_table = metadata.get("Function Table")
        if function_table:
            func_metadata = _create_function_metadata(row.tuple(), docstring, metadata)
            if function_table not in class_functions:
                class_functions[function_table] = []
            class_functions[function_table].append(func_metadata)
    return class_functions


def _create_function_metadata(row: tuple, docstring: str, metadata: dict[str, Any]  # type: ignore
) -> FunctionMetadata:
    function_type = metadata.get("Function Type", "free")
    function_class_values = _parse_function_class_values(
        metadata.get("Function Class Values", "")
    )
    arguments = _parse_arguments(row[3], function_class_values)

    return FunctionMetadata(
        schema=row[0],
        name=row[1],
        return_type=row[2],
        arguments=arguments,
        docstring=docstring,
        function_table=metadata["Function Table"],
        function_name=metadata.get("Function Name", row[1]),
        function_type=function_type,
        function_class_values=function_class_values,
    )


def _parse_comments(comment: str) -> tuple[str, dict[str, Any]]:
    if not comment:
        return "", {}

    docstring_match = re.search(r"DOCSTRING:\s*(.*?)\nMETADATA:", comment, re.DOTALL)
    metadata_match = re.search(r"METADATA:(.*)", comment, re.DOTALL)

    docstring = docstring_match.group(1).strip() if docstring_match else ""
    metadata: dict[str, Any] = {}

    if metadata_match:
        metadata_content = metadata_match.group(1).strip()
        for line in metadata_content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

    return docstring, metadata


def _parse_function_class_values(function_class_values_str: str) -> dict[str, str]:
    function_class_values = {}
    for arg in function_class_values_str.split(","):
        arg = arg.strip()
        if "=" in arg:
            func_arg, class_attr = arg.split("=")
            function_class_values[func_arg.strip()] = class_attr.strip()
    return function_class_values


def _parse_arguments(
    args_string: str, function_class_values: dict[str, str]
) -> list[ArgumentInfo]:
    arguments = []
    for arg in args_string.split(","):
        arg = arg.strip()
        if not arg:
            continue
        match = re.match(r"(\w+)\s+(\w+)(?:\s*=\s*(.+))?", arg)
        if match:
            name, type_, default = match.groups()
            class_sourced = name in function_class_values
            class_attribute = function_class_values.get(name, "")
            arguments.append(
                ArgumentInfo(
                    name=name,
                    type=type_,
                    default=default,
                    class_sourced=class_sourced,
                    class_attribute=class_attribute,
                )
            )
        else:
            raise ValueError(f"Could not parse argument: {arg}")
    return arguments
