from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from importlib.metadata import PackageNotFoundError, entry_points, version
from typing import TextIO

from sqlalchemy.engine import create_engine
from sqlalchemy.schema import MetaData

# Optional imports
optional_packages = ["citext", "geoalchemy2", "pgvector"]
for package in optional_packages:
    try:
        __import__(package)
        globals()[package] = True
    except ImportError:
        globals()[package] = False


def get_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    generators = {ep.name: ep for ep in entry_points(group="sqlacodegen.generators")}
    parser = argparse.ArgumentParser(
        description="Generates SQLAlchemy model code from an existing database."
    )
    parser.add_argument("url", nargs="?", help="SQLAlchemy url to the database")
    parser.add_argument(
        "--options", help="options (comma-delimited) passed to the generator class"
    )
    parser.add_argument(
        "--version", action="store_true", help="print the version number and exit"
    )
    parser.add_argument(
        "--schemas", help="load tables from the given schemas (comma-delimited)"
    )
    parser.add_argument(
        "--generator",
        choices=generators,
        default="aware",  # Default to aware generator
        help="generator class to use",
    )
    parser.add_argument(
        "--tables", help="tables to process (comma-delimited, default: all)"
    )
    parser.add_argument("--noviews", action="store_true", help="ignore views")
    parser.add_argument("--outfile", help="file to write output to (default: stdout)")
    args = parser.parse_args()

    if args.version:
        print(f"sqlacodegen version: {get_version('sqlacodegen')}")
        return

    if not args.url:
        print("You must supply a url\n", file=sys.stderr)
        parser.print_help()
        return

    for package in optional_packages:
        if globals()[package]:
            print(f"Using {package} {get_version(package)}")

    # Use reflection to fill in the metadata
    engine = create_engine(args.url)
    metadata = MetaData()
    tables = args.tables.split(",") if args.tables else None
    schemas = args.schemas.split(",") if args.schemas else [None]
    options = set(args.options.split(",")) if args.options else set()
    for schema in schemas:
        metadata.reflect(engine, schema, not args.noviews, tables)

    # Instantiate the generator
    generator_class = generators[args.generator].load()
    generator = generator_class(metadata, engine, options)

    # Open the target file (if given)
    with ExitStack() as stack:
        outfile: TextIO
        if args.outfile:
            outfile = open(args.outfile, "w", encoding="utf-8")
            stack.enter_context(outfile)
        else:
            outfile = sys.stdout

        # Write the generated model code to the specified file or standard output
        outfile.write(generator.generate())


if __name__ == "__main__":
    main()
