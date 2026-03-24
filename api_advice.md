The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

---

# PEP 8 – Style Guide for Python Code

Source: https://peps.python.org/pep-0008/

Authors: Guido van Rossum, Barry Warsaw, Alyssa Coghlan

## Key Principles

### A Foolish Consistency is the Hobgoblin of Little Minds

Code is read much more often than it is written. A style guide is about consistency:
- Consistency with the style guide is important
- Consistency within a project is more important
- Consistency within one module or function is the most important

Know when to be inconsistent – do not break backwards compatibility just to comply with style.

## Code Layout

- **Indentation**: Use 4 spaces per indentation level
- **Tabs or Spaces**: Spaces are preferred; never mix tabs and spaces
- **Maximum Line Length**: Limit lines to 79 characters (72 for docstrings/comments)
- **Blank Lines**: 2 blank lines around top-level definitions; 1 blank line around method definitions
- **Imports**: One import per line; group in order: standard library, related third party, local application

## Naming Conventions

- `module_name`, `package_name`
- `ClassName`, `ExceptionName`
- `function_name`, `method_name`, `variable_name`
- `CONSTANT_NAME`
- `_private`, `__mangled`

## Programming Recommendations

- Use `is` / `is not` for None comparisons, not `==`
- Use `isinstance()` instead of comparing types directly
- Use empty sequence truthiness: `if not seq:` not `if len(seq) == 0:`
- Be consistent in return statements
- Use `''.startswith()` and `''.endswith()` instead of slicing
- Use context managers (`with`) for resource management

## Type Annotations

- Use PEP 484 syntax for function annotations
- Single space after colon for variable annotations: `code: int`

---

*For the complete guide, see: https://peps.python.org/pep-0008/*
