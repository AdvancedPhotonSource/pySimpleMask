pyside6-uic mask.ui -o ../simplemask_ui.py
# pyuic5 -x mask.ui -o ../simplemask_ui.py

sed -i -e 's/\ pyqtgraph_mod/\ .pyqtgraph_mod/g' ../simplemask_ui.py
