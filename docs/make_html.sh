#!/bin/bash

# Make html
make clean
make html

# Postprocess the _static dir to static
python sphinxtogithub.py _build/html -v

# Rename all _sphinx_javascript_frameworks_compat.js to sphinx_javascript_frameworks_compat.js in all files in the static directory
find _build/html/static -type f -name "*.html" -exec sed -i 's/_sphinx_javascript_frameworks_compat.js/sphinx_javascript_frameworks_compat.js/g' {} \;
# Move _sphinx_javascript_frameworks_compat.js to sphinx_javascript_frameworks_compat.js in _build/html/static
mv _build/html/static/_sphinx_javascript_frameworks_compat.js _build/html/static/sphinx_javascript_frameworks_compat.js
# Print what was done
echo "- _sphinx_javascript_frameworks_compat.js renamed to sphinx_javascript_frameworks_compat.js"
