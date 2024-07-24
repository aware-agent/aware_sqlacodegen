from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from overrides import overrides
from typing import Any, Dict, List, Tuple
import black

from sqlalchemy import (
    Column,
    MetaData,
    Enum,
    Table,
    create_engine,
    text,
)
from sqlalchemy.engine import Connection, Engine

from ..models import (
    Model,
    ModelClass,
)

from ..utils import get_python_type
from ..models import RelationshipAttribute, RelationshipType, ColumnAttribute
from .sqlmodels import SQLModelGenerator


@dataclass
class ReverseRelationship:
    referencing_model: str
    foreign_key_field: str
    local_column: str
    target_column: str
    to_many: bool


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
        self.function_generator = FunctionGenerator(self.bind.engine.url)

    def generate(self) -> Any:
        unformatted_output = super().generate()

        def apply_black_formatter(code: str) -> str:
            mode = black.Mode(
                line_length=90,
                string_normalization=True,
                is_pyi=False,
            )
            return black.format_str(code, mode=mode)

        return apply_black_formatter(unformatted_output)

    def generate_models(self) -> list[Model]:
        models = super().generate_models()
        self.reverse_relationships = self.build_reverse_relationships(models)
        print("Debug - Reverse Relationships:")
        for table, relationships in self.reverse_relationships.items():
            print(f"  {table}:")
            for rel in relationships:
                print(
                    f"     {rel.referencing_model} -> {rel.foreign_key_field} - {rel.target_column} -> {rel.local_column} -- {rel.to_many}"
                )
        return models

    def collect_imports(self, models: Iterable[Model]) -> None:
        self.add_literal_import("__future__", "annotations")
        self.add_literal_import("typing", "Set, List")
        self.add_literal_import("pydantic", "Field")
        self.add_literal_import("datetime", "datetime")
        self.add_literal_import("aware_sql_python_types.base_model", "AwareSQLModel")
        self.add_literal_import(
            "aware_database_handlers.supabase.supabase_client_handler",
            "SupabaseClientHandler",
        )
        self.add_literal_import(
            "aware_database_handlers.supabase.utils.filter.filter", "Filter"
        )

    def build_reverse_relationships(
        self, models: List[Model]
    ) -> Dict[str, List[ReverseRelationship]]:
        reverse_relationships: Dict[str, List[ReverseRelationship]] = {}
        fk_cache: Dict[str, Dict[str, Tuple[str, str]]] = {}

        for model in models:
            if isinstance(model, ModelClass):
                fk_cache[model.name] = {
                    fk.column.table.name: (fk.parent.name, fk.column.name)
                    for fk in model.table.foreign_keys
                }

        for model in models:
            if isinstance(model, ModelClass):
                for relationship in model.relationships:
                    target_table = relationship.target.table.name
                    source_table = model.name

                    if target_table in fk_cache.get(source_table, {}):
                        local_column, target_column = fk_cache[source_table][target_table]

                        if target_table not in reverse_relationships:
                            reverse_relationships[target_table] = []

                        to_many = relationship.type in [
                            RelationshipType.MANY_TO_ONE,
                            RelationshipType.MANY_TO_MANY,
                        ]

                        reverse_relationships[target_table].append(
                            ReverseRelationship(
                                referencing_model=source_table,
                                foreign_key_field=local_column,
                                local_column=target_column,
                                target_column=local_column,
                                to_many=to_many,
                            )
                        )
                    else:
                        print(
                            f"Warning: Could not find foreign key for relationship {relationship.name} in {source_table}"
                        )

        return reverse_relationships

    def render_models(self, models: List[Model]) -> str:
        rendered: List[str] = []
        class_functions = self.function_generator.fetch_functions()

        for model in models:
            if isinstance(model, ModelClass):
                functions = class_functions.get(model.table.name, [])
                rendered.append(self.render_namespace(model, functions))

        return "\n".join(rendered)

    def render_namespace(
        self, model: ModelClass, functions: List[FunctionMetadata]
    ) -> str:
        rendered_class = self.render_class(model)
        rendered_class_methods = self.render_class_methods(model, functions)
        rendered_free_functions = self.render_free_functions(functions)
        namespace = "Ns" + model.name

        return f"""
class {namespace}:

{self.indent_all_lines(rendered_class)}

{self.indent_all_lines(rendered_class_methods)}

{self.indent_all_lines(rendered_free_functions)}
"""

    def render_class_properties(self, model: ModelClass) -> str:

        def render_forward_properties():
            # return []
            properties = []
            for fk in model.table.foreign_keys:
                target_table = fk.column.table
                property_name = self.to_snake_case(target_table.name)
                target_class = self.to_pascal_case(target_table.name)
                local_column = fk.parent.name
                target_column = fk.column.name

                property_def = f'''
@classmethod
def get_{property_name}(cls) -> Ns{target_class}.{target_class}:
{self.indentation}"""
{self.indentation}Forward Association to an Ns{target_class}.{target_class} instance.
{self.indentation}"""
{self.indentation}return Ns{target_class}.{target_class}.get("{target_column}", cls.{local_column})
'''
                properties.append(property_def)
            return properties

        def render_reverse_properties():
            properties = []
            if model.table.name in self.reverse_relationships:
                for reverse_rel in self.reverse_relationships[model.table.name]:
                    property_name = f"{self.to_snake_case(reverse_rel.referencing_model)}"
                    target_class = reverse_rel.referencing_model  # Keep original casing

                    if reverse_rel.to_many:
                        property_def = f'''
@classmethod
def get_{property_name}_list(cls, extra_filters: List[Filter] = []) -> List[Ns{target_class}.{target_class}]:
{self.indentation}"""
{self.indentation}Reverse referenced instances of Ns{target_class}.{target_class} with optional additional filters.
{self.indentation}"""
{self.indentation}return Ns{target_class}.{target_class}.get_list("{reverse_rel.target_column}", cls.{reverse_rel.local_column}, extra_filters)
'''
                    else:
                        property_def = f'''
@classmethod
def get_{property_name}(cls) -> Ns{target_class}.{target_class}:
{self.indentation}"""
{self.indentation}Reverse referenced instance of Ns{target_class}.{target_class}.
{self.indentation}"""
{self.indentation}return Ns{target_class}.{target_class}.get("{reverse_rel.target_column}", cls.{reverse_rel.local_column})
'''
                    properties.append(property_def)

            return properties

        forward_properties = render_forward_properties()
        reverse_properties = render_reverse_properties()

        all_properties = forward_properties + reverse_properties
        return "\n".join(all_properties)

    def to_snake_case(self, name: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def to_pascal_case(self, name: str) -> str:
        return "".join(word.capitalize() for word in name.split("_"))

    def render_class_methods(
        self, model: ModelClass, functions: List[FunctionMetadata]
    ) -> str:
        class_methods = [
            self.render_function(model, func)
            for func in functions
            if func.function_type == "class"
        ]
        return "\n".join(class_methods)

    def render_class(self, model: ModelClass) -> str:
        class_definition = super().render_class(model)
        class_definition.replace(f"class {model.name}(", f"class {model.name}(", 1)
        class_definition += self.indent_all_lines(self.render_class_properties(model))
        return class_definition

    def render_functions(
        self, model: ModelClass, functions: List[FunctionMetadata]
    ) -> str:
        return "\n".join(self.render_function(model, func) for func in functions)

    def render_function(self, model: ModelClass | None, func: FunctionMetadata) -> str:
        if func.function_type == "class":
            if model is None:
                raise ValueError("Model is required for class methods")
            self.handle_function_arguments(model, func)
            return self.render_class_method(func, model.name)
        else:
            return self.render_free_function(func)

    def handle_function_arguments(self, model: ModelClass, func: FunctionMetadata):
        column_names = set(column.name for column in model.columns)
        for arg in func.arguments:
            if arg.name in column_names:
                arg.class_sourced = True
                arg.class_attribute = f"self.{arg.name}"

    def render_class_method(self, func: FunctionMetadata, class_name: str) -> str:
        return self.render_rpc_method(func, is_class_method=True)

    def render_free_function(self, func: FunctionMetadata) -> str:
        return self.render_rpc_method(func, is_class_method=False)

    def render_free_functions(self, functions: List[FunctionMetadata]) -> str:
        free_functions = [
            self.render_function(None, func)
            for func in functions
            if func.function_type != "class"
        ]
        return "\n".join(free_functions)

    def render_rpc_method(
        self,
        func_meta: FunctionMetadata,
        is_class_method: bool = True,
    ) -> str:
        python_name = func_meta.function_name or func_meta.name
        args_str = self.render_arguments(func_meta.arguments)
        return_python_type = get_python_type(func_meta.return_type)
        docstring = self.render_docstring(func_meta, return_python_type)
        rpc_args_str = self.render_rpc_arguments(func_meta.arguments)

        if is_class_method:
            return self.render_class_rpc_method(
                python_name,
                args_str,
                return_python_type,
                docstring,
                func_meta.name,
                rpc_args_str,
            )
        else:
            return self.render_free_rpc_method(
                python_name,
                args_str,
                return_python_type,
                docstring,
                func_meta.name,
                rpc_args_str,
            )

    def render_arguments(self, arguments):
        return ", ".join(
            f"{arg.name}: {get_python_type(arg.type)}"
            for arg in arguments
            if not arg.class_sourced
        )

    def render_rpc_arguments(self, arguments):
        rpc_args = []
        for arg in arguments:
            if arg.class_sourced:
                rpc_arg = f'"{arg.name}": self.{arg.class_attribute}'
            else:
                rpc_arg = f'"{arg.name}": {arg.name}'
            rpc_args.append(rpc_arg)
        return ",\n".join(f"{self.indentation * 4}{arg}" for arg in rpc_args)

    def render_class_rpc_method(
        self, name, args, return_type, docstring, rpc_name, rpc_args
    ):
        indented_docstring = "\n".join(
            f"{self.indentation * 2}{line}" for line in docstring.split("\n")
        )
        method = f"""
{self.indentation}def {name}(self, {args}) -> {return_type}:
{indented_docstring}
{self.indentation *2}return SupabaseClientHandler().get_supabase_client().rpc(
{self.indentation *2}"{rpc_name}",
        {{
{rpc_args}
        }}
    ).execute()
"""
        return method

    def render_free_rpc_method(
        self, name, args, return_type, docstring, rpc_name, rpc_args
    ):
        indented_docstring = self.indent_all_lines(docstring)
        return f"""
@staticmethod
def {name}({args}) -> {return_type}:
{indented_docstring}
{self.indentation}return SupabaseClientHandler().get_supabase_client().rpc(
{self.indentation * 2}"{rpc_name}",
        {{
{rpc_args}
        }}
{self.indentation}).execute()
"""

    def render_docstring(self, func_meta: FunctionMetadata, return_type: str) -> str:
        filtered_docstring = self.filter_docstring(func_meta)
        return f'''"""
{filtered_docstring}
{self.indentation}Returns:
{self.indentation * 2}{return_type}
"""
'''

    def filter_docstring(self, func_meta: FunctionMetadata) -> str:
        lines = func_meta.docstring.split("\n")
        filtered_lines = []
        param_section = False
        for line in lines:
            if "Parameters:" in line:
                param_section = True
            if not (
                param_section
                and any(
                    arg.name in line for arg in func_meta.arguments if arg.class_sourced
                )
            ):
                filtered_lines.append(line)
        return "\n".join(f"{line}" for line in filtered_lines)

    @overrides
    def render_relationship(self, relationship: RelationshipAttribute) -> str:
        return ""

        relationship.target_ns = f"Ns{relationship.target.name}"
        relationship.enable_private = False
        # relationship.prefix_private = True
        relationship.rename_lists = True
        return super().render_relationship(relationship)

    def render_table_args(self, table: Table) -> str:
        return ""

    def render_module_variables(self, models: list[Model]) -> str:
        return ""

    def render_class_declaration(self, model: ModelClass) -> str:
        if model.parent_class:
            parent = model.parent_class.name
        else:
            parent = self.base_class_name

        superclass_part = f"({parent})"
        return f"class {model.name}{superclass_part}:"

    def indent_all_lines(self, s):
        return "\n".join(f"{self.indentation}{line}" for line in s.split("\n"))

    def render_column_attribute(self, column_attr: ColumnAttribute) -> str:
        column = column_attr.column
        try:
            python_type = column.type.python_type
        except NotImplementedError:
            print(f"Warning: Could not determine Python type for column {column.name}", column.type)
            python_type_name = "Any"
        else:
            python_type_name = python_type.__name__

        # Translate UUID to str to comply with Pydantic
        if python_type_name == "UUID":
            python_type_name = "str"

        kwargs: dict[str, Any] = {}
        if (
            column.autoincrement and column.name in column.table.primary_key
        ) or column.nullable:
            self.add_literal_import("typing", "Optional")
            kwargs["default"] = None
            python_type_name = f"Optional[{python_type_name}]"

        rendered_column_field = self.render_column(column, True, is_table=True)
        #return rendered_column
        
        #kwargs["sa_column"] = f"{rendered_column}"
        #rendered_field = render_callable("Field", kwargs=kwargs)
        return f"{column_attr.name}: {python_type_name} = {rendered_column_field}"

    def render_column(
        self, column: Column[Any], show_name: bool, is_table: bool = False
    ) -> str:
        name = column.name if show_name else "_"
        
        python_type = get_python_type(str(column.type))
        
        field_args = []
        field_kwargs = {}
        
        # Handle Enum types
        if isinstance(column.type, Enum):
            python_type = self.process_enum_type(column.type)
        
        # Handle default values
        if column.default is not None:
            if column.default.is_scalar:
                field_kwargs['default'] = repr(column.default.arg)
            else:
                field_kwargs['default'] = 'None'
        elif column.server_default is not None or not column.nullable:
            field_args.append('...')  # Pydantic's way to indicate a required field
        
        # Handle nullable
        if not column.nullable and not column.primary_key:
            field_kwargs['nullable'] = 'False'
        
        # Handle primary key
        if column.primary_key:
            field_kwargs['primary_key'] = 'True'
        
        # Handle comment/description
        if hasattr(column, 'comment') and column.comment:
            field_kwargs['description'] = repr(column.comment)
        
        # Generate the Field definition
        args_str = ', '.join(field_args)
        kwargs_str = ', '.join(f'{k}={v}' for k, v in field_kwargs.items())
        all_args = ', '.join(filter(None, [args_str, kwargs_str]))
        
        return f"Field({kwargs_str})" if field_kwargs else "None"
        
    def process_enum_type(self, coltype: Enum) -> str:
        if coltype.name is not None:
            # Generate a Python-style class name from the Enum name
            enum_class_name = "".join(
                word.capitalize() for word in coltype.name.split("_")
            )
            # Add the enum class name and its values to the set
            self.enums_to_generate.add((enum_class_name, tuple(coltype.enums)))
            return enum_class_name
        return "str"  # Default to str if no name is provided

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
    arguments: List[ArgumentInfo]
    docstring: str = ""
    function_table: str = ""
    function_name: str = ""
    function_type: str = "free"
    function_class_values: Dict[str, str] = field(default_factory=dict)


class FunctionGenerator:
    def __init__(self, engine_url):
        self.engine = create_engine(engine_url)

    def fetch_functions(self) -> Dict[str, List[FunctionMetadata]]:
        query = text(self.get_function_query())
        with self.engine.connect() as conn:
            result = conn.execute(query)
            return self.process_query_results(result)

    def get_function_query(self):
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

    def process_query_results(self, result) -> Dict[str, List[FunctionMetadata]]:
        class_functions: Dict[str, List[FunctionMetadata]] = {}
        for row in result.fetchall():
            docstring, metadata = self.parse_comments(row[4])
            function_table = metadata.get("Function Table")
            if function_table:
                func_metadata = self.create_function_metadata(row, docstring, metadata)
                if function_table not in class_functions:
                    class_functions[function_table] = []
                class_functions[function_table].append(func_metadata)
        return class_functions

    def create_function_metadata(
        self, row: tuple, docstring: str, metadata: Dict[str, Any]  # type: ignore
    ) -> FunctionMetadata:
        function_type = metadata.get("Function Type", "free")
        function_class_values = self.parse_function_class_values(
            metadata.get("Function Class Values", "")
        )
        arguments = self.parse_arguments(row[3], function_class_values)

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

    def parse_comments(self, comment: str) -> tuple[str, Dict[str, Any]]:
        if not comment:
            return "", {}

        docstring_match = re.search(r"DOCSTRING:\s*(.*?)\nMETADATA:", comment, re.DOTALL)
        metadata_match = re.search(r"METADATA:(.*)", comment, re.DOTALL)

        docstring = docstring_match.group(1).strip() if docstring_match else ""
        metadata: Dict[str, Any] = {}

        if metadata_match:
            metadata_content = metadata_match.group(1).strip()
            for line in metadata_content.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

        return docstring, metadata

    def parse_function_class_values(
        self, function_class_values_str: str
    ) -> Dict[str, str]:
        function_class_values = {}
        for arg in function_class_values_str.split(","):
            arg = arg.strip()
            if "=" in arg:
                func_arg, class_attr = arg.split("=")
                function_class_values[func_arg.strip()] = class_attr.strip()
        return function_class_values

    def parse_arguments(
        self, args_string: str, function_class_values: Dict[str, str]
    ) -> List[ArgumentInfo]:
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
