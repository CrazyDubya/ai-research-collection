import os
import ast
import astor
import argparse
import uuid
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class FunctionDefTransformer(ast.NodeTransformer):
    """
    Transformer to modify function definitions:
    - Add 'name_suffix' parameter.
    - Insert 'person.define("name_suffix", name_suffix)' in the function body.
    """
    def __init__(self):
        super().__init__()
        self.modified = False

    def visit_FunctionDef(self, node):
        if node.name.startswith("create_"):
            # Check if 'name_suffix' is already a parameter
            if not any(arg.arg == "name_suffix" for arg in node.args.args):
                # Add 'name_suffix' as a parameter
                node.args.args.append(ast.arg(arg='name_suffix', annotation=None))
                self.modified = True
                logging.info(f"Added 'name_suffix' parameter to function '{node.name}'.")

                # Create 'person.define("name_suffix", name_suffix)' statement
                define_call = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='person', ctx=ast.Load()),
                            attr='define',
                            ctx=ast.Load()
                        ),
                        args=[
                            ast.Str(s='name_suffix'),
                            ast.Name(id='name_suffix', ctx=ast.Load())
                        ],
                        keywords=[]
                    )
                )

                # Insert the define_call at the beginning of the function body
                node.body.insert(0, define_call)
                logging.info(f"Inserted 'person.define(\"name_suffix\", name_suffix)' in function '{node.name}'.")
        return self.generic_visit(node)

class FunctionCallTransformer(ast.NodeTransformer):
    """
    Transformer to modify function calls:
    - Add 'name_suffix=unique_id' keyword argument.
    """
    def __init__(self, unique_id):
        super().__init__()
        self.unique_id = unique_id
        self.modified = False

    def visit_Call(self, node):
        # Check if the function being called is a 'create_*' function
        if isinstance(node.func, ast.Name) and node.func.id.startswith("create_"):
            # Check if 'name_suffix' is already a keyword argument
            if not any(kw.arg == "name_suffix" for kw in node.keywords):
                # Add 'name_suffix=unique_id' as a keyword argument
                kw = ast.keyword(arg='name_suffix', value=ast.Name(id='unique_id', ctx=ast.Load()))
                node.keywords.append(kw)
                self.modified = True
                logging.info(f"Appended 'name_suffix=unique_id' to call '{node.func.id}()'.")
        return self.generic_visit(node)

def migrate_file(file_path, unique_id):
    """
    Migrate a single Python file:
    - Modify function definitions to accept 'name_suffix'.
    - Modify function calls to pass 'name_suffix=unique_id'.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        source = file.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logging.error(f"Syntax error in file {file_path}: {e}")
        return False, 0

    # Transform function definitions
    func_def_transformer = FunctionDefTransformer()
    tree = func_def_transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # Transform function calls
    func_call_transformer = FunctionCallTransformer(unique_id)
    tree = func_call_transformer.visit(tree)
    ast.fix_missing_locations(tree)

    if func_def_transformer.modified or func_call_transformer.modified:
        # Backup the original file
        backup_path = f"{file_path}.bak"
        shutil.copyfile(file_path, backup_path)
        logging.info(f"Backup created for {file_path} at {backup_path}.")

        # Generate the new source code
        new_source = astor.to_source(tree)

        # Write the modified code back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_source)
        logging.info(f"File {file_path} has been modified.")

        # Count the number of modifications
        modifications = 0
        if func_def_transformer.modified:
            modifications += 1
        if func_call_transformer.modified:
            modifications += 1
        return True, modifications
    return False, 0

def migrate(directory, unique_id):
    """
    Walk through the specified directory and migrate all Python files.
    """
    modified_files_count = 0
    total_modifications = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                modified, mods = migrate_file(file_path, unique_id)
                if modified:
                    modified_files_count += 1
                    total_modifications += mods
    return modified_files_count, total_modifications

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Migrate Python files to include name_suffix parameter in create_* functions.')
    parser.add_argument('-d', '--directory', required=True, help='Root directory to start the migration')
    args = parser.parse_args()

    # Generate a unique ID
    unique_id = "unique_id"  # This should be defined in trouoin.py

    # Inform the user about the migration process
    logging.info(f"Starting migration in directory: {args.directory}")

    # Perform migration
    count, mods = migrate(args.directory, unique_id)
    logging.info(f"Migrated {count} files with a total of {mods} modifications.")

