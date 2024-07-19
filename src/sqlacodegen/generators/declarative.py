from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pprint import pformat
from textwrap import indent
from typing import Any, ClassVar

import inflect
from sqlalchemy import (
    ForeignKeyConstraint,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    UniqueConstraint,
)
from sqlalchemy.engine import Connection, Engine

from ..models import (
    ColumnAttribute,
    JoinType,
    Model,
    ModelClass,
    RelationshipAttribute,
    RelationshipType,
)
from ..utils import (
    get_column_names,
    get_common_fk_constraints,
    get_constraint_sort_key,
    qualified_table_name,
    render_callable,
    uses_default_name,
)


from ..models import Base, LiteralImport
from ..utils import re_invalid_identifier
from .tables import TablesGenerator


class DeclarativeGenerator(TablesGenerator):
    valid_options: ClassVar[set[str]] = TablesGenerator.valid_options | {
        "use_inflect",
        "nojoined",
        "nobidi",
    }

    def __init__(
        self,
        metadata: MetaData,
        bind: Connection | Engine,
        options: Sequence[str],
        *,
        indentation: str = "    ",
        base_class_name: str = "Base",
    ):
        super().__init__(metadata, bind, options, indentation=indentation)
        self.base_class_name: str = base_class_name
        self.inflect_engine = inflect.engine()

    def generate_base(self) -> None:
        self.base = Base(
            literal_imports=[LiteralImport("sqlalchemy.orm", "DeclarativeBase")],
            declarations=[
                f"class {self.base_class_name}(DeclarativeBase):",
                f"{self.indentation}pass",
            ],
            metadata_ref=f"{self.base_class_name}.metadata",
        )

    def collect_imports(self, models: Iterable[Model]) -> None:
        super().collect_imports(models)
        if any(isinstance(model, ModelClass) for model in models):
            self.add_literal_import("sqlalchemy.orm", "Mapped")
            self.add_literal_import("sqlalchemy.orm", "mapped_column")

    def collect_imports_for_model(self, model: Model) -> None:
        super().collect_imports_for_model(model)
        if isinstance(model, ModelClass):
            if model.relationships:
                self.add_literal_import("sqlalchemy.orm", "relationship")

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

        # Change base if we only have tables
        if not any(
            isinstance(model, ModelClass) for model in models_by_table_name.values()
        ):
            super().generate_base()

        # Collect the imports
        self.collect_imports(models_by_table_name.values())

        # Rename models and their attributes that conflict with imports or other
        # attributes
        global_names = {name for namespace in self.imports.values() for name in namespace}
        for model in models_by_table_name.values():
            self.generate_model_name(model, global_names)
            global_names.add(model.name)

        return list(models_by_table_name.values())

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

                # Generate the opposite end of the relationship in the target class
                if "nobidi" not in self.options:
                    if r_type is RelationshipType.MANY_TO_ONE:
                        r_type = RelationshipType.ONE_TO_MANY

                    reverse_relationship = RelationshipAttribute(
                        r_type,
                        target,
                        source,
                        constraint,
                        foreign_keys=relationship.foreign_keys,
                        backref=relationship,
                    )
                    relationship.backref = reverse_relationship
                    target.relationships.append(reverse_relationship)

                    # For self referential relationships, remote_side needs to be set
                    if source is target:
                        reverse_relationship.remote_side = [
                            source.get_column_attribute(colname)
                            for colname in constraint.column_keys
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
                if "nobidi" not in self.options:
                    reverse_relationship = RelationshipAttribute(
                        RelationshipType.MANY_TO_MANY,
                        target,
                        source,
                        fk_constraints[0],
                        association_table,
                        relationship,
                    )
                    relationship.backref = reverse_relationship
                    target.relationships.append(reverse_relationship)

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
            super().generate_model_name(model, global_names)

    def generate_column_attr_name(
        self,
        column_attr: ColumnAttribute,
        global_names: set[str],
        local_names: set[str],
    ) -> None:
        column_attr.name = self.find_free_name(
            column_attr.column.name, global_names, local_names
        )

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

            if "use_inflect" in self.options:
                if relationship.type in (
                    RelationshipType.ONE_TO_MANY,
                    RelationshipType.MANY_TO_MANY,
                ):
                    inflected_name = self.inflect_engine.plural_noun(preferred_name)
                    if inflected_name:
                        preferred_name = inflected_name
                else:
                    inflected_name = self.inflect_engine.singular_noun(preferred_name)
                    if inflected_name:
                        preferred_name = inflected_name

        relationship.name = self.find_free_name(preferred_name, global_names, local_names)

    def render_models(self, models: list[Model]) -> str:
        rendered: list[str] = []
        for model in models:
            if isinstance(model, ModelClass):
                rendered.append(self.render_class(model))
            else:
                rendered.append(f"{model.name} = {self.render_table(model.table)}")

        return "\n\n\n".join(rendered)

    def render_class(self, model: ModelClass) -> str:
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

    def render_class_declaration(self, model: ModelClass) -> str:
        parent_class_name = (
            model.parent_class.name if model.parent_class else self.base_class_name
        )
        return f"class {model.name}({parent_class_name}):"

    def render_class_variables(self, model: ModelClass) -> str:
        variables = [f"__tablename__ = {model.table.name!r}"]

        # Render constraints and indexes as __table_args__
        table_args = self.render_table_args(model.table)
        if table_args:
            variables.append(f"__table_args__ = {table_args}")

        return "\n".join(variables)

    def render_table_args(self, table: Table) -> str:
        args: list[str] = []
        kwargs: dict[str, str] = {}

        # Render constraints
        for constraint in sorted(table.constraints, key=get_constraint_sort_key):
            if uses_default_name(constraint):
                if isinstance(constraint, PrimaryKeyConstraint):
                    continue
                if (
                    isinstance(constraint, (ForeignKeyConstraint, UniqueConstraint))
                    and len(constraint.columns) == 1
                ):
                    continue

            args.append(self.render_constraint(constraint))

        # Render indexes
        for index in sorted(table.indexes, key=lambda i: i.name):
            if len(index.columns) > 1 or not uses_default_name(index):
                args.append(self.render_index(index))

        if table.schema:
            kwargs["schema"] = table.schema

        if table.comment:
            kwargs["comment"] = table.comment

        if kwargs:
            formatted_kwargs = pformat(kwargs)
            if not args:
                return formatted_kwargs
            else:
                args.append(formatted_kwargs)

        if args:
            rendered_args = f",\n{self.indentation}".join(args)
            if len(args) == 1:
                rendered_args += ","

            return f"(\n{self.indentation}{rendered_args}\n)"
        else:
            return ""

    def render_column_attribute(self, column_attr: ColumnAttribute) -> str:
        column = column_attr.column
        rendered_column = self.render_column(column, column_attr.name != column.name)

        try:
            python_type = column.type.python_type
            python_type_name = python_type.__name__
            if python_type.__module__ == "builtins":
                column_python_type = python_type_name
            else:
                python_type_module = python_type.__module__
                column_python_type = f"{python_type_module}.{python_type_name}"
                self.add_module_import(python_type_module)
        except NotImplementedError:
            self.add_literal_import("typing", "Any")
            column_python_type = "Any"

        if column.nullable:
            self.add_literal_import("typing", "Optional")
            column_python_type = f"Optional[{column_python_type}]"
        return f"{column_attr.name}: Mapped[{column_python_type}] = {rendered_column}"

    def render_relationship(self, relationship: RelationshipAttribute) -> str:
        def render_column_attrs(column_attrs: list[ColumnAttribute]) -> str:
            rendered = []
            for attr in column_attrs:
                if attr.model is relationship.source:
                    rendered.append(attr.name)
                else:
                    rendered.append(repr(f"{attr.model.name}.{attr.name}"))

            return "[" + ", ".join(rendered) + "]"

        def render_foreign_keys(column_attrs: list[ColumnAttribute]) -> str:
            rendered = []
            render_as_string = False
            # Assume that column_attrs are all in relationship.source or none
            for attr in column_attrs:
                if attr.model is relationship.source:
                    rendered.append(attr.name)
                else:
                    rendered.append(f"{attr.model.name}.{attr.name}")
                    render_as_string = True

            if render_as_string:
                return "'[" + ", ".join(rendered) + "]'"
            else:
                return "[" + ", ".join(rendered) + "]"

        def render_join(terms: list[JoinType]) -> str:
            rendered_joins = []
            for source, source_col, target, target_col in terms:
                rendered = f"lambda: {source.name}.{source_col} == {target.name}."
                if target.__class__ is Model:
                    rendered += "c."

                rendered += str(target_col)
                rendered_joins.append(rendered)

            if len(rendered_joins) > 1:
                rendered = ", ".join(rendered_joins)
                return f"and_({rendered})"
            else:
                return rendered_joins[0]

        # Render keyword arguments
        kwargs: dict[str, Any] = {}
        if relationship.type is RelationshipType.ONE_TO_ONE and relationship.constraint:
            if relationship.constraint.referred_table is relationship.source.table:
                kwargs["uselist"] = False

        # Add the "secondary" keyword for many-to-many relationships
        if relationship.association_table:
            table_ref = relationship.association_table.table.name
            if relationship.association_table.schema:
                table_ref = f"{relationship.association_table.schema}.{table_ref}"

            kwargs["secondary"] = repr(table_ref)

        if relationship.remote_side:
            kwargs["remote_side"] = render_column_attrs(relationship.remote_side)

        if relationship.foreign_keys:
            kwargs["foreign_keys"] = render_foreign_keys(relationship.foreign_keys)

        if relationship.primaryjoin:
            kwargs["primaryjoin"] = render_join(relationship.primaryjoin)

        if relationship.secondaryjoin:
            kwargs["secondaryjoin"] = render_join(relationship.secondaryjoin)

        if relationship.backref:
            kwargs["back_populates"] = repr(relationship.backref.name)

        rendered_relationship = render_callable(
            "relationship", repr(relationship.target.name), kwargs=kwargs
        )

        relationship_type: str
        if relationship.type == RelationshipType.ONE_TO_MANY:
            self.add_literal_import("typing", "List")
            relationship_type = f"List['{relationship.target.name}']"
        elif relationship.type in (
            RelationshipType.ONE_TO_ONE,
            RelationshipType.MANY_TO_ONE,
        ):
            relationship_type = f"'{relationship.target.name}'"
        elif relationship.type == RelationshipType.MANY_TO_MANY:
            self.add_literal_import("typing", "List")
            relationship_type = f"List['{relationship.target.name}']"
        else:
            self.add_literal_import("typing", "Any")
            relationship_type = "Any"

        return (
            f"{relationship.name}: Mapped[{relationship_type}] "
            f"= {rendered_relationship}"
        )
