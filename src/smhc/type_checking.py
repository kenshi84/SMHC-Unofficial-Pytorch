import os

from beartype import beartype
from jaxtyping import jaxtyped

# check the environment variable
TYPECHECK_DISABLED = os.environ.get("DISABLE_TYPES", "0").lower() in (
    "1",
    "true",
    "yes",
)

def check_types(func):
    """unified decorator that applies optional type-checking"""
    if TYPECHECK_DISABLED:
        return func

    # wrap the function with beartype checker
    return jaxtyped(typechecker=beartype)(func)
