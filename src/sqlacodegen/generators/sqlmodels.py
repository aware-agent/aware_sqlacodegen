from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, List, Tuple, Dict

from sqlalchemy import (
    Column,
    MetaData,
)
from sqlalchemy.engine import Connection, Engine

from ..models import (
    ColumnAttribute,
    Model,
    ModelClass,
    RelationshipAttribute,
    RelationshipType,
)
from ..utils import (
    get_column_names,
    get_constraint_sort_key,
    qualified_table_name,
    render_callable,
)

from ..models import Base
from .tables import TablesGenerator
from .declarative import DeclarativeGenerator


class SQLModelGenerator(DeclarativeGenerator):
    def __init__(
        self,
        metadata: MetaData,
        bind: Connection | Engine,
        options: Sequence[str],
        *,
        indentation: str = "    ",
        base_class_name: str = "SQLModel",
    ):
        super().__init__(
            metadata,
            bind,
            options,
            indentation=indentation,
            base_class_name=base_class_name,
        )

    def generate_models(self) -> list[Model]:
        models_by_table_name: dict[str, Model] = {}

        # Pick association tables from the metadata into their own set, don't process
        # them normally
        links: defaultdict[str, list[Model]] = defaultdict(lambda: [])
        for table in self.metadata.sorted_tables:
            qualified_name = qualified_table_name(table)

            # Link tables have exactly two foreign key constraints and all columns are
            # involved in them
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

            # Only form model classes for tables that have a primary key and are not
            # association tables
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

        # Change base if we have both tables and model classes
        if any(
            not isinstance(model, ModelClass) for model in models_by_table_name.values()
        ):
            TablesGenerator.generate_base(self)

        # Collect the imports
        self.collect_imports(models_by_table_name.values())

        # Rename models and their attributes that conflict with imports or other
        # attributes
        global_names = {name for namespace in self.imports.values() for name in namespace}
        for model in models_by_table_name.values():
            self.generate_model_name(model, global_names)
            global_names.add(model.name)

        return list(models_by_table_name.values())

    def generate_base(self) -> None:
        self.base = Base(
            literal_imports=[],
            declarations=[],
            metadata_ref="",
        )

    def collect_imports(self, models: Iterable[Model]) -> None:
        super(DeclarativeGenerator, self).collect_imports(models)
        if any(isinstance(model, ModelClass) for model in models):
            self.add_literal_import("sqlmodel", "SQLModel")
            self.add_literal_import("sqlmodel", "Field")

    def collect_imports_for_model(self, model: Model) -> None:
        super(DeclarativeGenerator, self).collect_imports_for_model(model)
        if isinstance(model, ModelClass):
            for column_attr in model.columns:
                if column_attr.column.nullable:
                    self.add_literal_import("typing", "Optional")
                    break

            if model.relationships:
                self.add_literal_import("sqlmodel", "Relationship")

            for relationship_attr in model.relationships:
                if relationship_attr.type in (
                    RelationshipType.ONE_TO_MANY,
                    RelationshipType.MANY_TO_MANY,
                ):
                    self.add_literal_import("typing", "List")

    def collect_imports_for_column(self, column: Column[Any]) -> None:
        super().collect_imports_for_column(column)
        try:
            python_type = column.type.python_type
        except NotImplementedError:
            self.add_literal_import("typing", "Any")
        else:
            self.add_import(python_type)

    def render_module_variables(self, models: list[Model]) -> str:
        declarations: list[str] = self.base.declarations
        if any(not isinstance(model, ModelClass) for model in models):
            if self.base.table_metadata_declaration is not None:
                declarations.append(self.base.table_metadata_declaration)

        return "\n".join(declarations)

    def render_class_declaration(self, model: ModelClass) -> str:
        if model.parent_class:
            parent = model.parent_class.name
        else:
            parent = self.base_class_name

        superclass_part = f"({parent}, table=True)"
        return f"class {model.name}{superclass_part}:"

    def render_class_variables(self, model: ModelClass) -> str:
        variables = []

        if model.table.name != model.name.lower():
            variables.append(f"__tablename__ = {model.table.name!r}")

        # Render constraints and indexes as __table_args__
        table_args = self.render_table_args(model.table)
        if table_args:
            variables.append(f"__table_args__ = {table_args}")

        return "\n".join(variables)

    def render_column_attribute(self, column_attr: ColumnAttribute) -> str:
        column = column_attr.column
        try:
            python_type = column.type.python_type
        except NotImplementedError:
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

        rendered_column = self.render_column(column, True, is_table=True)
        kwargs["sa_column"] = f"{rendered_column}"
        rendered_field = render_callable("Field", kwargs=kwargs)
        return f"{column_attr.name}: {python_type_name} = {rendered_field}"

    def render_relationship(self, relationship: RelationshipAttribute) -> str:
        rendered = super().render_relationship(relationship).partition(" = ")[2]
        args, is_upward, is_to_many = self.render_relationship_args(
            rendered, relationship
        )
        if not relationship.enable_upwards and is_upward:
            return ""

        kwargs: Dict[str, Any] = {}

        annotation = repr(
            (f"{relationship.target_ns}.{relationship.target.name}")
            if relationship.target_ns
            else relationship.target.name
        )

        if is_to_many:
            self.add_literal_import("typing", "List")
            annotation = f"List[{annotation}]"
        else:
            self.add_literal_import("typing", "Optional")
            annotation = f"Optional[{annotation}]"

        rendered_field = render_callable("Relationship", *args, kwargs=kwargs)

        if relationship.rename_lists and is_to_many:
            relationship_name = relationship.name + "_list"
        else:
            relationship_name = relationship.name

        return f"{relationship_name}: {annotation} = {rendered_field}"

    def render_relationship_args(
        self, arguments: str, relationship: RelationshipAttribute
    ) -> Tuple[List[str], bool, bool]:
        argument_list = arguments.split(",")
        argument_list[-1] = argument_list[-1].rstrip(") ")
        argument_list = [argument.strip() for argument in argument_list]

        rendered_args: List[str] = []
        back_populates_value = None
        is_to_many = relationship.type in (
            RelationshipType.ONE_TO_MANY,
            RelationshipType.MANY_TO_MANY,
        )

        for arg in argument_list:
            if "back_populates" in arg:
                rendered_args.append(arg)
                back_populates_value = arg.split("=")[1].strip().strip("'\"")
            elif "uselist=False" in arg:
                rendered_args.append("sa_relationship_kwargs={'uselist': False}")

        # Determine if it's an upward relationship
        current_table = relationship.source.table.name
        target_table = relationship.target.table.name

        is_upward = (back_populates_value == current_table) or (
            current_table == target_table
        )

        return rendered_args, is_upward, is_to_many
