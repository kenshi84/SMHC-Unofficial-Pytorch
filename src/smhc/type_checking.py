import os

from jaxtyping import jaxtyped
from rich.console import Console

# check the environment variable
TYPECHECK_DISABLED = os.environ.get("DISABLE_TYPES", "0").lower() in (
    "1",
    "true",
    "yes",
)


def initialize_type_checker():
    """selecting the type checker based on the input or installed libraries"""
    # completely bypass typechecking for maximum performance
    # however at current version of the code, I have added some manual rudimentary checks for some of the function calls!
    if TYPECHECK_DISABLED:
        Console().print(
            "[yellow]smhc: disabled type checking (via DISABLE_TYPES=1) ...[/yellow]"
        )
        return None

    try:
        # try to load the strict typeguard checker first (used in dev mode)
        from typeguard import typechecked as typechecker

        Console().print(
            "[yellow]smhc: strict type checking using typeguard ...[/yellow]"
        )

    except ImportError:
        # fallback to the fast beartype chcker
        # since beartype is in main dependencies, this will always be available
        from beartype import beartype as typechecker

        Console().print("[yellow]smhc: fast type checking using beartype ...[/yellow]")

    return typechecker


SELECTED_TYPECHECKER = initialize_type_checker()


def check_types(func):
    """unified decorator that applies the chosen type-checking strategy."""
    if SELECTED_TYPECHECKER is None:
        return func

    # wrap the function with the chosen checker
    return jaxtyped(typechecker=SELECTED_TYPECHECKER)(func)
