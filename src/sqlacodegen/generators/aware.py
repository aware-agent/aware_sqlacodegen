from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import (
    MetaData,
    create_engine,
    text,
)
from sqlalchemy.engine import Connection, Engine

from ..models import (
    Model,
    ModelClass,
)

from ..utils import get_python_type, re_invalid_identifier
from .sqlmodels import SQLModelGenerator


class AwareGenerator(SQLModelGenerator):
    def __init__(
        self,
        metadata: MetaData,
        bind: Connection | Engine,
        options: Sequence[str],
        *,
        indentation: str = "    ",
        base_class_name: str = "AwareSQLModel",
    ):
        super().__init__(
            metadata,
            bind,
            options,
            indentation=indentation,
            base_class_name=base_class_name,
        )

    def collect_imports(self, models: Iterable[Model]) -> None:
        super().collect_imports(models)
        self.remove_literal_import("sqlmodel", "SQLModel")
        self.add_literal_import("aware_sql_python_types.base_model", "AwareSQLModel")
        self.add_literal_import("typing", "Set")

    def generate_base(self) -> None:
        self.function_generator = FunctionGenerator(self.bind.engine.url)
        super().generate_base()

    def generate_model_name(self, model: Model, global_names: set[str]) -> None:
        if isinstance(model, ModelClass):
            preferred_name = re_invalid_identifier.sub("_", model.table.name)
            preferred_name = "".join(
                part[:1].upper() + part[1:] for part in preferred_name.split("_")
            )
            if "use_inflect" in self.options:
                singular_name = self.inflect_engine.singular_noun(preferred_name)
                if singular_name:
                    preferred_name = singular_name

            preferred_name = f"Pg{preferred_name}"
            model.name = self.find_free_name(preferred_name, global_names)

            local_names: set[str] = set()
            for column_attr in model.columns:
                self.generate_column_attr_name(column_attr, global_names, local_names)
                local_names.add(column_attr.name)

            for relationship in model.relationships:
                self.generate_relationship_name(relationship, global_names, local_names)
                local_names.add(relationship.name)
        else:
            super().generate_model_name(model, global_names)

    def render_models(self, models: list[Model]) -> str:
        rendered: list[str] = []
        class_functions = self.function_generator.fetch_functions()
        for model in models:
            if isinstance(model, ModelClass):
                functions = class_functions.get(model.table.name, [])
                rendered.append(self.render_class_with_supabase_methods(model, functions))
            else:
                rendered.append(f"{model.name} = {self.render_table(model.table)}")

        return "\n\n\n".join(rendered)

    def render_class_with_supabase_methods(
        self, model: ModelClass, functions: list[FunctionMetadata]
    ) -> str:
        rendered_class = self.render_class(model)

        rendered_methods = []
        for func in functions:
            rpc_method = self.render_rpc_method(func)
            rendered_methods.append(rpc_method)

        rendered_class += "\n" + "\n".join(rendered_methods)

        return rendered_class

    def parse_argument(self, arg: str) -> tuple[str, str, str | None]:
        pattern = r"^(\w+)\s+([\w\s\[\]]+?)(?:\s+DEFAULT\s+(.*))?$"
        match = re.match(pattern, arg, re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not parse argument: {arg}")
        name, type_name, default = match.groups()

        python_type = get_python_type(type_name)
        if default:
            default = re.sub(r"::[\w\[\]]+", "", default)
            is_array = "[]" in type_name
            if is_array:
                default = default.replace("ARRAY", "").strip("[]")
                default = f"[{default}]"
            if default == "NULL":
                default = "None"
                python_type = f"Optional[{python_type}]"
        return name, python_type, default

    def render_rpc_method(self, func_meta: FunctionMetadata) -> str:
        python_name = func_meta.function_name or func_meta.name

        arg_strings = [
            f"{arg.name}: {get_python_type(arg.type)}"
            for arg in func_meta.arguments
            if not arg.class_sourced
        ]
        args_str = ", ".join(arg_strings)

        return_python_type = get_python_type(func_meta.return_type)

        # Generate docstring using render_docstring
        docstring = self.render_docstring(func_meta, return_python_type)

        rpc_args = [
            f'"{arg.name}": '
            f'{"self." + arg.class_attribute if arg.class_sourced else arg.name}'
            for arg in func_meta.arguments
        ]

        rpc_args_str = ",\n".join(f"{self.indentation * 4}{arg}" for arg in rpc_args)

        method = f"""
{self.indentation}def {python_name}(self, {args_str}) -> {return_python_type}:
{docstring}
{self.indentation * 2}return self.get_supabase_client().rpc(
{self.indentation * 3}"{func_meta.name}",
{self.indentation * 3}{{
{rpc_args_str}
{self.indentation * 3}}}
{self.indentation * 2}).execute().data"""
        return method

    def render_docstring(self, func_meta: FunctionMetadata, return_type: str) -> str:
        filtered_docstring = self.filter_docstring(func_meta)

        docstring = (
            f'{self.indentation * 2}"""\n{self.indentation}{filtered_docstring}\n\n'
        )
        docstring += (
            f"{self.indentation * 2}Returns:\n{self.indentation * 2}    {return_type}\n"
        )
        docstring += f'{self.indentation * 2}"""'
        return docstring

    def filter_docstring(self, func_meta: FunctionMetadata) -> str:
        lines = func_meta.docstring.split("\n")
        filtered_lines = []
        param_section = False
        for line in lines:
            if "Parameters:" in line:
                param_section = True
            if param_section and any(
                arg.name in line for arg in func_meta.arguments if arg.class_sourced
            ):
                continue
            filtered_lines.append(line)
        return "\n".join(f"{self.indentation}{line}" for line in filtered_lines)


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
    class_sourced_args: dict[str, str] = field(default_factory=dict)


class FunctionGenerator:
    def __init__(self, engine_url: str):
        self.engine = create_engine(engine_url)

    def fetch_functions(self) -> dict[str, list[FunctionMetadata]]:
        query = text(
            """
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
        )
        with self.engine.connect() as conn:
            result = conn.execute(query)
            class_functions: dict[str, list[FunctionMetadata]] = {}
            for row in result.fetchall():
                docstring, metadata = self.parse_comments(row[4])
                function_table = metadata.get("Function Table")
                if function_table:
                    class_sourced_args = self.parse_class_sourced_args(
                        metadata.get("Class Sourced Args", "")
                    )
                    arguments = self.parse_arguments(row[3], class_sourced_args)
                    func_metadata = FunctionMetadata(
                        schema=row[0],
                        name=row[1],
                        return_type=row[2],
                        arguments=arguments,
                        docstring=docstring,
                        function_table=function_table,
                        function_name=metadata.get("Function Name", row[1]),
                        class_sourced_args=class_sourced_args,
                    )
                    if function_table not in class_functions:
                        class_functions[function_table] = []
                    class_functions[function_table].append(func_metadata)
            return class_functions

    def parse_comments(self, comment: str) -> tuple[str, dict[str, Any]]:
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

    def parse_class_sourced_args(self, class_sourced_args_str: str) -> dict[str, str]:
        class_sourced_args = {}
        for arg in class_sourced_args_str.split(","):
            arg = arg.strip()
            if "=" in arg:
                func_arg, class_attr = arg.split("=")
                class_sourced_args[func_arg.strip()] = class_attr.strip()
        return class_sourced_args

    def parse_arguments(
        self, args_string: str, class_sourced_args: dict[str, str]
    ) -> list[ArgumentInfo]:
        arguments = []
        for arg in args_string.split(","):
            arg = arg.strip()
            if not arg:
                continue
            match = re.match(r"(\w+)\s+(\w+)(?:\s*=\s*(.+))?", arg)
            if match:
                name, type_, default = match.groups()
                class_sourced = name in class_sourced_args
                class_attribute = class_sourced_args.get(name, "")
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
