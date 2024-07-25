from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from overrides import overrides
from typing import Any, Dict, List, Tuple
import black
from textwrap import indent

from ..utils import (
    get_column_names,
    get_compiled_expression,
    get_python_type,
    qualified_table_name,
    get_common_fk_constraints,
)

from sqlalchemy import (
    Column,
    MetaData,
    Enum,
    DefaultClause,
    Computed,
    Table,
    FetchedValue,
    ClauseElement,
    Dialect,
    TextClause,
    create_engine,
    text,
)
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.type_api import TypeEngine

from ..models import (
    Model,
    ModelClass,
)

from ..utils import get_python_type
from ..models import RelationshipAttribute, RelationshipType, ColumnAttribute

from pydantic import Field

from .functions import fetch_functions, FunctionMetadata
from typing import ClassVar
from sys import builtin_module_names

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
from typing import ClassVar

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

Imports = dict[str, set[str]]  # type alias


@dataclass
class EnumInfo:
    name: str
    values: list[str]

    def __hash__(self):
        hash_value = hash(self.name)
        return hash_value

    def __eq__(self, other):
        if not isinstance(other, EnumInfo):
            return NotImplemented
        equality = self.name == other.name
        return equality


@dataclass
class ReverseRelationship:
    referencing_model: str
    foreign_key_field: str
    local_column: str
    target_column: str
    to_many: bool


@dataclass
class Generator():
    metadata: MetaData
    bind: Connection | Engine
    options: Sequence[str] = field(default_factory=list)
    indentation: str = "    "
    base_class_name: str = "AwareSQLModel"
    valid_options: ClassVar[set[str]] = {
        "nojoined",
    }
    builtin_module_names: ClassVar[set[str]] = set(__import__('sys').builtin_module_names) | {"dataclasses"}
    imports: ClassVar[dict[str, set[str]]] = defaultdict(set)
    module_imports: ClassVar[set[str]] = set()

    def generate(self) -> Any:
        unformatted_output = self._generate()

        unformatted_resolved_output = self.resolve_unresolved_types(unformatted_output)

        def apply_black_formatter(code: str) -> str:
            mode = black.Mode(
                line_length=90,
                string_normalization=True,
                is_pyi=False,
            )
            return black.format_str(code, mode=mode)

        return apply_black_formatter(unformatted_resolved_output)

    def add_literal_import(self, pkgname: str, name: str) -> None:
        names = self.imports.setdefault(pkgname, set())
        names.add(name)

    def _generate(self) -> str:
        sections: list[str] = []

        def should_ignore_table(table: Table) -> bool:
            # Support for Alembic and sqlalchemy-migrate -- never expose the schema version tables
            return table.name in ("alembic_version", "migrate_version")

        def fix_column_types(table: Table) -> None:
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
                                    table.c[colname].type = Enum(
                                        *options, native_enum=False
                                    )

                                continue

            def get_adapted_type(coltype: TypeEngine[Any], dialect: Dialect) -> any:
                compiled_type = coltype.compile(dialect)
                for supercls in coltype.__class__.__mro__:
                    if not supercls.__name__.startswith("_") and hasattr(
                        supercls, "__visit_name__"
                    ):
                        # Hack to fix adaptation of the Enum class which is broken since
                        # SQLAlchemy 1.2
                        kw = {}
                        if supercls is Enum:
                            kw["name"] = coltype.name
                            # TODO : i think we can remove this

                        try:
                            new_coltype = coltype.adapt(supercls)
                            # TODO : i think we can remove this
                        except TypeError:
                            # If the adaptation fails, don't try again
                            break

                        for key, value in kw.items():
                            setattr(new_coltype, key, value)

                        if isinstance(coltype, ARRAY):
                            new_coltype.item_type = get_adapted_type(
                                new_coltype.item_type, dialect
                            )

                        try:
                            # If the adapted column type does not render the same as the
                            # original, don't substitute it
                            if (
                                new_coltype.compile(self.bind.engine.dialect)
                                != compiled_type
                            ):
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

            for column in table.c:
                try:
                    column.type = get_adapted_type(
                        coltype=column.type, dialect=self.bind.engine.dialect
                    )
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
                                column.default = sqlalchemy.Sequence(
                                    seqname, schema=schema
                                )

                            column.server_default = None

        def render_python_enum(name: str, values: list[str]) -> str:
            enum_members = "\n    ".join(
                [f"{value.upper()} = '{value}'" for value in values]
            )
            self.add_literal_import("enum", "Enum")
            return f"class {name}(str, Enum):\n    {enum_members}\n"

        # Remove unwanted elements from the metadata
        for table in list(self.metadata.tables.values()):
            if should_ignore_table(table):
                self.metadata.remove(table)
                continue

        # Use information from column constraints to figure out the intended column
        # types
        for table in self.metadata.tables.values():
            fix_column_types(table)

        # Generate the models
        models: list[Model] = self.generate_models()

        # Render module level variables
        variables = self.render_module_variables(models)
        if variables:
            sections.append(variables + "\n")

        # Render models
        self.enums_to_generate: set[EnumInfo] = set()
        rendered_models = self.render_models(models)
        if rendered_models:
            sections.append(rendered_models)

        # Render enums
        for enum_info in self.enums_to_generate:
            rendered_enum = render_python_enum(enum_info.name, enum_info.values)
            sections.insert(0, rendered_enum)  # Insert enums at the beginning

        # Render collected imports
        groups = self.group_imports()
        imports = "\n\n".join("\n".join(line for line in group) for group in groups)
        if imports:
            sections.insert(0, imports)

        return "\n\n".join(sections) + "\n"

    def resolve_unresolved_types(self, source: str) -> str:
        def replace_unresolved(match):
            unresolved_type = self.to_pascal_case(match.group(1))

            # Try to match with enum name directly
            for enum_info in self.enums_to_generate:
                if enum_info.name == unresolved_type:
                    return enum_info.name

            # If still no match, raise an error
            raise ValueError(f"Could not resolve type: {unresolved_type}")

        unresolved_pattern = re.compile(r"UNRESOLVED_(\w+)_ENUM")
        return unresolved_pattern.sub(replace_unresolved, source)

    def generate_models(self) -> list[Model]:
        models = self._generate_models()
        self.reverse_relationships = self.build_reverse_relationships(models)
        return models

    def _generate_models(self) -> list[Model]:
        models_by_table_name: dict[str, Model] = {}

        # Pick association tables from the metadata into their own set, don't process them normally
        links: defaultdict[str, list[Model]] = defaultdict(lambda: [])
        for table in self.metadata.sorted_tables:
            qualified_name = qualified_table_name(table)

            # Link tables have exactly two foreign key constraints and all columns are involved in them
            fk_constraints = sorted(
                table.foreign_key_constraints, key=get_constraint_sort_key
            )
            if len(fk_constraints) == 2 and all(
                col.foreign_keys for col in table.columns
            ):
                model = models_by_table_name[qualified_name] = Model(table)
                tablename = fk_constraints[0].elements[0].column.table.name
                links[tablename].append(model)
                continue

            # Only form model classes for tables that have a primary key and are not association tables
            if not table.primary_key:
                models_by_table_name[qualified_name] = Model(table)
            else:
                model = ModelClass(table)
                models_by_table_name[qualified_name] = model

                # Fill in the columns
                for column in table.c:
                    column_attr = ColumnAttribute(model, column)
                    model.columns.append(column_attr)

        # Add relationships
        for model in models_by_table_name.values():
            if isinstance(model, ModelClass):
                self.generate_relationships(
                    model, models_by_table_name, links[model.table.name]
                )

        # Nest inherited classes in their superclasses to ensure proper ordering
        if "nojoined" not in self.options:
            for model in list(models_by_table_name.values()):
                if not isinstance(model, ModelClass):
                    continue

                pk_column_names = {col.name for col in model.table.primary_key.columns}
                for constraint in model.table.foreign_key_constraints:
                    if set(get_column_names(constraint)) == pk_column_names:
                        target = models_by_table_name[
                            qualified_table_name(constraint.elements[0].column.table)
                        ]
                        if isinstance(target, ModelClass):
                            model.parent_class = target
                            target.children.append(model)

        # Collect the imports
        self.collect_imports()

        # Rename models and their attributes that conflict with imports or other
        # attributes
        global_names = {name for namespace in self.imports.values() for name in namespace}
        for model in models_by_table_name.values():
            self.generate_model_name(model, global_names)
            global_names.add(model.name)

        return list(models_by_table_name.values())

    def generate_model_name(self, model: Model, global_names: set[str]) -> None:
        if isinstance(model, ModelClass):
            preferred_name = re_invalid_identifier.sub("_", model.table.name)
            preferred_name = "".join(
                part[:1].upper() + part[1:] for part in preferred_name.split("_")
            )

            model.name = self.find_free_name(preferred_name, global_names)

            # Fill in the names for column attributes
            local_names: set[str] = set()
            for column_attr in model.columns:
                self.generate_column_attr_name(column_attr, global_names, local_names)
                local_names.add(column_attr.name)

            # Fill in the names for relationship attributes
            for relationship in model.relationships:
                self.generate_relationship_name(relationship, global_names, local_names)
                local_names.add(relationship.name)
        else:
            preferred_name = f"t_{model.table.name}"
            model.name = self.find_free_name(preferred_name, global_names)

    # UTILS !

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

    def generate_column_attr_name(
        self,
        column_attr: ColumnAttribute,
        global_names: set[str],
        local_names: set[str],
    ) -> None:
        column_attr.name = self.find_free_name(
            column_attr.column.name, global_names, local_names
        )

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

    def generate_relationship_name(
        self,
        relationship: RelationshipAttribute,
        global_names: set[str],
        local_names: set[str],
    ) -> None:
        # Self referential reverse relationships
        preferred_name: str
        if (
            relationship.type
            in (RelationshipType.ONE_TO_MANY, RelationshipType.ONE_TO_ONE)
            and relationship.source is relationship.target
            and relationship.backref
            and relationship.backref.name
        ):
            preferred_name = relationship.backref.name + "_reverse"
        else:
            preferred_name = relationship.target.table.name

            # If there's a constraint with a single column that ends with "_id", use the
            # preceding part as the relationship name
            if relationship.constraint:
                is_source = relationship.source.table is relationship.constraint.table
                if is_source or relationship.type not in (
                    RelationshipType.ONE_TO_ONE,
                    RelationshipType.ONE_TO_MANY,
                ):
                    column_names = [c.name for c in relationship.constraint.columns]
                    if len(column_names) == 1 and column_names[0].endswith("_id"):
                        preferred_name = column_names[0][:-3]

        relationship.name = self.find_free_name(preferred_name, global_names, local_names)

    # END OF UTILS ?

    def generate_relationships(
        self,
        source: ModelClass,
        models_by_table_name: dict[str, Model],
        association_tables: list[Model],
    ) -> list[RelationshipAttribute]:
        relationships: list[RelationshipAttribute] = []
        reverse_relationship: RelationshipAttribute | None

        # Add many-to-one (and one-to-many) relationships
        pk_column_names = {col.name for col in source.table.primary_key.columns}
        for constraint in sorted(
            source.table.foreign_key_constraints, key=get_constraint_sort_key
        ):
            target = models_by_table_name[
                qualified_table_name(constraint.elements[0].column.table)
            ]
            if isinstance(target, ModelClass):
                if "nojoined" not in self.options:
                    if set(get_column_names(constraint)) == pk_column_names:
                        parent = models_by_table_name[
                            qualified_table_name(constraint.elements[0].column.table)
                        ]
                        if isinstance(parent, ModelClass):
                            source.parent_class = parent
                            parent.children.append(source)
                            continue

                # Add uselist=False to One-to-One relationships
                column_names = get_column_names(constraint)
                if any(
                    isinstance(c, (PrimaryKeyConstraint, UniqueConstraint))
                    and {col.name for col in c.columns} == set(column_names)
                    for c in constraint.table.constraints
                ):
                    r_type = RelationshipType.ONE_TO_ONE
                else:
                    r_type = RelationshipType.MANY_TO_ONE

                relationship = RelationshipAttribute(r_type, source, target, constraint)
                source.relationships.append(relationship)

                # For self referential relationships, remote_side needs to be set
                if source is target:
                    relationship.remote_side = [
                        source.get_column_attribute(col.name)
                        for col in constraint.referred_table.primary_key
                    ]

                # If the two tables share more than one foreign key constraint,
                # SQLAlchemy needs an explicit primaryjoin to figure out which column(s)
                # it needs
                common_fk_constraints = get_common_fk_constraints(
                    source.table, target.table
                )
                if len(common_fk_constraints) > 1:
                    relationship.foreign_keys = [
                        source.get_column_attribute(key) for key in constraint.column_keys
                    ]

        # Add many-to-many relationships
        for association_table in association_tables:
            fk_constraints = sorted(
                association_table.table.foreign_key_constraints,
                key=get_constraint_sort_key,
            )
            target = models_by_table_name[
                qualified_table_name(fk_constraints[1].elements[0].column.table)
            ]
            if isinstance(target, ModelClass):
                relationship = RelationshipAttribute(
                    RelationshipType.MANY_TO_MANY,
                    source,
                    target,
                    fk_constraints[1],
                    association_table,
                )
                source.relationships.append(relationship)

                # Generate the opposite end of the relationship in the target class
                reverse_relationship = None

                # Add a primary/secondary join for self-referential many-to-many
                # relationships
                if source is target:
                    both_relationships = [relationship]
                    reverse_flags = [False, True]
                    if reverse_relationship:
                        both_relationships.append(reverse_relationship)

                    for relationship, reverse in zip(both_relationships, reverse_flags):
                        if (
                            not relationship.association_table
                            or not relationship.constraint
                        ):
                            continue

                        constraints = sorted(
                            relationship.constraint.table.foreign_key_constraints,
                            key=get_constraint_sort_key,
                            reverse=reverse,
                        )
                        pri_pairs = zip(
                            get_column_names(constraints[0]),
                            constraints[0].elements,
                        )
                        sec_pairs = zip(
                            get_column_names(constraints[1]),
                            constraints[1].elements,
                        )
                        relationship.primaryjoin = [
                            (
                                relationship.source,
                                elem.column.name,
                                relationship.association_table,
                                col,
                            )
                            for col, elem in pri_pairs
                        ]
                        relationship.secondaryjoin = [
                            (
                                relationship.target,
                                elem.column.name,
                                relationship.association_table,
                                col,
                            )
                            for col, elem in sec_pairs
                        ]

        return relationships

    def collect_imports(self) -> None:
        self.add_literal_import("__future__", "annotations")
        self.add_literal_import("typing", "Set, List")
        self.add_literal_import("uuid", "UUID")
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

        return reverse_relationships

    def render_models(self, models: List[Model]) -> str:
        rendered: List[str] = []
        class_functions = fetch_functions(self.bind.engine.url)

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
        class_definition = self._render_class(model)
        class_definition.replace(f"class {model.name}(", f"class {model.name}(", 1)
        class_definition += self.indent_all_lines(self.render_class_properties(model))
        return class_definition

    def _render_class(self, model: ModelClass) -> str:
        sections: list[str] = []

        # Render class variables / special declarations
        class_vars: str = self.render_class_variables(model)
        if class_vars:
            sections.append(class_vars)

        # Render column attributes
        rendered_column_attributes: list[str] = []
        for nullable in (False, True):
            for column_attr in model.columns:
                if column_attr.column.nullable is nullable:
                    rendered_column_attributes.append(
                        self.render_column_attribute(column_attr)
                    )

        if rendered_column_attributes:
            sections.append("\n".join(rendered_column_attributes))

        # Render relationship attributes
        rendered_relationship_attributes: list[str] = [
            self.render_relationship(relationship) for relationship in model.relationships
        ]

        if rendered_relationship_attributes:
            sections.append("\n".join(rendered_relationship_attributes))

        declaration = self.render_class_declaration(model)
        rendered_sections = "\n\n".join(
            indent(section, self.indentation) for section in sections
        )
        return f"{declaration}\n{rendered_sections}"

    def render_class_variables(self, model: ModelClass) -> str:
        variables = []

        if model.table.name != model.name.lower():
            variables.append(f"__tablename__ = {model.table.name!r}")

        # Render constraints and indexes as __table_args__
        table_args = self.render_table_args(model.table)
        if table_args:
            variables.append(f"__table_args__ = {table_args}")

        return "\n".join(variables)

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

    def resolve_type(self, type_: str) -> str:
        python_type = get_python_type(type_)
        if not python_type:
            python_type = "UNRESOLVED_" + type_ + "_ENUM"

        return python_type

    def render_rpc_method(
        self,
        func_meta: FunctionMetadata,
        is_class_method: bool = True,
    ) -> str:
        python_name = func_meta.function_name or func_meta.name
        args_str = self.render_arguments(func_meta.arguments)
        return_python_type = self.resolve_type(func_meta.return_type)
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
            f"{arg.name}: {self.resolve_type(arg.type)}"
            for arg in arguments
            if not arg.class_sourced
        )

    def render_rpc_arguments(self, arguments):
        rpc_args = []
        for arg in arguments:

            python_type = self.resolve_type(arg.type)
            assume_enum = False
            if python_type.startswith("UNRESOLVED_") and python_type.endswith("_ENUM"):
                assume_enum = True

            if arg.class_sourced:
                rpc_arg = f'"{arg.name}": self.{arg.class_attribute}'
            else:
                rpc_arg = f'"{arg.name}": {arg.name}'

            if assume_enum:
                rpc_arg += ".value"

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

    def render_relationship(self, relationship: RelationshipAttribute) -> str:
        return ""

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
        python_type = column.type.python_type
        python_type_name = python_type.__name__

        kwargs: dict[str, Any] = {}
        if (
            column.autoincrement and column.name in column.table.primary_key
        ) or column.nullable:
            self.add_literal_import("typing", "Optional")
            kwargs["default"] = None
            python_type_name = f"Optional[{python_type_name}]"

        rendered_column_field = self.render_column(column, True, is_table=True)
        var_base = f"{column_attr.name}: {python_type_name}"

        if rendered_column_field:
            var_base += f" = {rendered_column_field}"
        return var_base

    def render_column(
        self, column: Column[Any], show_name: bool, is_table: bool = False
    ) -> str:
        name = column.name if show_name else "_"

        field_args = []
        field_kwargs = {}

        # Handle Enum types
        if isinstance(column.type, Enum):
            _ = self.process_enum_type(column.type)

        # Handle default values
        if column.server_default:
            if column.server_default.has_argument and isinstance(
                column.server_default, DefaultClause
            ):
                if isinstance(column.server_default.arg, str):
                    field_kwargs["default"] = repr(column.server_default.arg)

        if column.nullable is False:
            field_args.append("...")

        # Handle comment/description
        if hasattr(column, "comment") and column.comment:
            field_kwargs["description"] = repr(column.comment)

        # Generate the Field definition
        args_str = ", ".join(field_args)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in field_kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        return f"Field({all_args})" if all_args else ""

    def process_enum_type(self, coltype: Enum) -> str:
        if coltype.name is None:
            raise ValueError("Enum type must have a name")
        enum_class_name = self.to_pascal_case(coltype.name)
        # Create and add the new enum
        new_enum = EnumInfo(name=enum_class_name, values=list(coltype.enums))
        self.enums_to_generate.add(new_enum)
        return enum_class_name
